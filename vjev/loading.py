"""Load a vjev model from the Hugging Face Hub or a local directory, on CUDA, Apple silicon
or CPU. A model repo holds the full merged weights (a Qwen3.5 image-text model, bf16), the
scoring head (head.pt), and vjev.json describing how inputs are rendered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import ListwiseScorer, pause_token_id
from .render import linear_patch_embed


def unit_lower_inverse(A: torch.Tensor) -> torch.Tensor:
    """(I + L)^-1 for L = the strictly lower triangle of A (..., n, n), by matmuls only.

    Block recursion, bottom-up: for T = [[P, 0], [C, Q]],  T^-1 = [[P^-1, 0], [-Q^-1 C P^-1, Q^-1]].
    Starting from 1x1 blocks and doubling, n = 64 takes 6 levels of two small batched
    matmuls; every intermediate is the true inverse of a sub-block, so it is as well behaved
    as forward substitution. (The finite Neumann product (I - L)(I + L^2)(I + L^4)... is
    exact on paper and broke on the real model: near-identical neighbouring keys push L
    towards the all-ones triangle, whose powers reach ~1e18 and cancel into NaN.)"""
    n = A.size(-1)
    m = 1 << max(n - 1, 0).bit_length()                  # next power of two
    L = A.tril(-1)
    if m != n:
        L = torch.nn.functional.pad(L, (0, m - n, 0, m - n))
    batch = L.shape[:-2]
    inv = torch.ones(*batch, m, 1, 1, dtype=A.dtype, device=A.device)      # m blocks of 1x1
    s = 1
    while s < m:
        nb = m // (2 * s)
        D = torch.diagonal(L.reshape(*batch, nb, 2 * s, nb, 2 * s), dim1=-4, dim2=-2).movedim(-1, -3)
        C = D[..., s:, :s]
        P, Q = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
        low = -(Q @ C @ P)
        top = torch.cat([P, torch.zeros_like(P)], dim=-1)
        inv = torch.cat([top, torch.cat([low, Q], dim=-1)], dim=-2)
        s *= 2
    return inv.squeeze(-3)[..., :n, :n]


def patch_mps_triangular_solve() -> None:
    """Route unit-lower triangular solves on MPS through unit_lower_inverse.

    transformers' reference gated delta rule (the only one without CUDA) calls
    torch.linalg.solve_triangular twice per linear-attention layer. On MPS (torch 2.9, M1
    Pro) one call took 0.52 s against 0.003 s for a matmul of the same shape - those two
    calls were 72% of the whole forward. Every other device keeps the real solver."""
    real = torch.linalg.solve_triangular
    if getattr(real, "_vjev_patched", False):
        return

    def solve(A, B, *, upper, left=True, unitriangular=False, out=None):
        if A.device.type == "mps" and unitriangular and not upper and left and out is None:
            return unit_lower_inverse(A) @ B
        return real(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)

    solve._vjev_patched = True
    torch.linalg.solve_triangular = solve


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Loaded:
    model: ListwiseScorer
    processor: object            # AutoProcessor: tokenizer + image processor
    config: object               # the model config (vision token ids, text_config)
    meta: dict                   # vjev.json
    model_id: str
    device: torch.device


def load(repo: str | Path, device: str | None = None, dtype=None) -> Loaded:
    """`repo`: a Hub id such as "yah01/vjev-vision-pilot", or a local directory with the
    same files. bf16 on CUDA; fp16 on MPS (M1-family GPUs have no native bf16, and fp16
    tracks fp32 more closely there); fp32 on CPU."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or pick_device()
    if device == "mps":
        patch_mps_triangular_solve()
    dtype = dtype or {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)
    local = Path(repo).is_dir()
    get = (lambda f: Path(repo) / f) if local else (lambda f: Path(hf_hub_download(str(repo), f)))
    meta = json.loads(get("vjev.json").read_text(encoding="utf-8"))
    if meta.get("arch", "vl") != "vl":
        raise ValueError(f"{repo} is a text-only checkpoint ({meta.get('arch')}); this package "
                         f"serves the image-text models")

    processor = AutoProcessor.from_pretrained(str(repo))
    full = AutoModelForImageTextToText.from_pretrained(str(repo), dtype=dtype, attn_implementation="sdpa")
    trunk = full.model                            # the vocabulary projection is unused
    trunk.config.use_cache = False
    linear_patch_embed(trunk.visual)
    trunk.config.pause_id = pause_token_id(full.config)
    scorer = ListwiseScorer(trunk, full.config.text_config.hidden_size, int(meta.get("pause", 0)))
    scorer.load_heads(get("head.pt").parent, device="cpu")
    scorer.to(device).eval()
    return Loaded(scorer, processor, full.config, meta,
                  model_id=str(repo).rstrip("/").split("/")[-1], device=torch.device(device))

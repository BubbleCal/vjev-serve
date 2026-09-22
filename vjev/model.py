"""Listwise scorer: every option scored in ONE forward pass, by ONE shared head.

Why this shape
--------------
Scoring each option in a separate forward pass and normalising afterwards is
*mathematically* guaranteed to satisfy IIA: adding or removing an option cannot change
the ratio between two others. The Jev API, measured, violates that - adding a
semantically competing option costs the others a paired delta of +0.027 over the
permutation noise floor, and the leader is protected. A pointwise scorer scores exactly
0.0 on both. So the options have to see each other, which means one sequence.

Three constraints fall out of the data and the API:

  one shared head        Option count runs 2..227 in our corpus and Jev allows 255. A
                         `num_labels=n` classification head cannot span that, let alone
                         extrapolate. A single `w·h + b` applied to every option slot is
                         independent of n by construction.

  no new special tokens  Qwen3.5 has a 248,320-entry *tied* embedding, so resizing pays
                         twice and perturbs the output distribution. Slots are read at
                         ordinary tokens: the restated labels after the option list
                         ("Answer: (A) (B) ...", render.VisionRenderer readout="trailing").

  soft targets           Trained on whole probability distributions (KL to the teacher,
                         soft BCE for noul), never on an argmax: calibration is the point.

Where the slots sit matters more than it first looked. Read at each option's own last
token (readout="inline"), a causal mask means an option can never see the options after
it, so an appended option leaves every earlier logit bit-identical and the model is
pointwise-plus-renormalisation after all (measured: appended options left earlier logits bit-identical). The
trailing readout puts every slot after the whole list; order dependence then remains only
among the restated labels, comparable to the teacher's own (TVD ~0.05 on the teacher, ~0.035 here).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

CHOICE, NOUL, SCORE = 0, 1, 2
QTYPE_NAMES = {CHOICE: "choice", NOUL: "noul", SCORE: "score"}


@dataclass
class Batch:
    """A padded batch. `slots[b, k]` indexes the last token of option k in row b."""
    input_ids: torch.Tensor          # (B, S) int64
    attention_mask: torch.Tensor     # (B, S) int64
    slots: torch.Tensor              # (B, K) int64, -1 where padded
    slot_mask: torch.Tensor          # (B, K) bool
    target: torch.Tensor             # (B, K) float; noul uses column 0 only
    qtype: torch.Tensor              # (B,) int64
    shards: tuple[str, ...] = ()     # per-row origin, for per-shard loss attribution
    # Extra trunk inputs for image rows (pixel_values, image_grid_thw, mm_token_type_ids;
    # see vision.collate_mm). None for text-only batches, which therefore run unchanged.
    mm: dict | None = None

    def to(self, device: str | torch.device) -> "Batch":
        move = lambda v: v.to(device) if isinstance(v, torch.Tensor) else v
        fields = {k: move(v) for k, v in self.__dict__.items() if k != "mm"}
        return Batch(**fields, mm={k: move(v) for k, v in self.mm.items()} if self.mm else None)

    @property
    def n_tokens(self) -> int:
        return int(self.attention_mask.sum())


def pause_token_id(config) -> int:
    """The placeholder id pause tokens travel under: the last row of the embedding table.
    Qwen3.5's table has 248,320 rows and its tokenizer 248,077 entries, so the rows past
    the tokenizer are never produced from text - nothing a caller sends can collide with
    it, and the table is not resized (the constraint in the module docstring)."""
    return getattr(config, "text_config", config).vocab_size - 1


class PauseTokens(nn.Module):
    """K learned input embeddings, spliced in wherever `pause_id` appears (an experiment; every released checkpoint has 0).

    The model emits no text, so its only serial budget is one forward pass. K extra
    positions between the option list and the readout give it K more columns of compute
    and KV scratch space before it has to commit. They train straight from the listwise
    loss - the slots attend to them - so there is no thought supervision to invent.

    Spliced in by a hook on the embedding layer rather than by passing inputs_embeds: the
    multimodal trunk locates image placeholders and builds M-RoPE positions from
    input_ids, and must keep receiving them.
    """

    def __init__(self, k: int, hidden_size: int, pause_id: int, std: float = 0.02):
        super().__init__()
        self.pause_id = pause_id
        self.emb = nn.Parameter(torch.randn(k, hidden_size) * std)      # fp32, like the head

    def splice(self, _module, args, kwargs, out):
        ids = args[0] if args else kwargs["input"]
        m = ids == self.pause_id
        if not m.any():
            return None
        which = (m.cumsum(1) - 1)[m]              # a row's pause tokens, in order: 0..K-1
        if int(which.max()) >= self.emb.size(0):
            raise ValueError(f"a row carries {int(which.max()) + 1} pause tokens, "
                             f"the model has {self.emb.size(0)}")
        out = out.clone()
        out[m] = self.emb[which].to(out.dtype)
        return out


class ListwiseScorer(nn.Module):
    """Transformer trunk + one shared linear head over option slots.

    The trunk is the bare model (no vocab projection): this head replaces the 248,320-way
    lm_head. Note that dropping lm_head frees no *weights* — it is tied to
    the input embedding — but it does remove a (B, S, 248320) logit tensor per step.
    """

    def __init__(self, trunk: nn.Module, hidden_size: int, n_pause: int = 0):
        super().__init__()
        self.trunk = trunk
        self.pause: PauseTokens | None = None
        if n_pause:
            emb = trunk.get_input_embeddings()
            # start at the scale of real token embeddings, measured rather than assumed
            std = float(emb.weight[:50_000].float().std())
            self.pause = PauseTokens(n_pause, hidden_size, pause_token_id(trunk.config), std)
            emb.register_forward_hook(self.pause.splice, with_kwargs=True)
        # fp32: ~2.5k parameters, kept exact whatever the trunk runs in.
        self.head = nn.Linear(hidden_size, 1, dtype=torch.float32)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, input_ids, attention_mask, slots, slot_mask, **mm) -> torch.Tensor:
        """-> (B, K) logits, -inf on padded slots. `mm` carries the multimodal trunk
        inputs for image rows and is empty for text, so the text path is unchanged."""
        h = self.trunk(input_ids=input_ids, attention_mask=attention_mask, **mm).last_hidden_state
        idx = slots.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.size(-1))
        gathered = h.gather(1, idx).float()                       # (B, K, d)
        logits = self.head(gathered).squeeze(-1)                  # (B, K)
        return logits.masked_fill(~slot_mask, float("-inf"))

    def load_heads(self, d: Path, device: str = "cpu", fresh_pause_ok: bool = False) -> None:
        """head.pt, and pause.pt when the checkpoint has pause tokens."""
        d = Path(d)
        if not (d / "head.pt").exists():
            raise FileNotFoundError(f"no head.pt in {d} - the trunk alone predicts nothing")
        self.head.load_state_dict(torch.load(d / "head.pt", map_location=device))
        has = (d / "pause.pt").exists()
        if has and self.pause is None:
            raise ValueError(f"{d} was trained with pause tokens; build the model with the count in vjev.json")
        if self.pause is not None:
            if has:
                self.pause.load_state_dict(torch.load(d / "pause.pt", map_location=device))
            elif not fresh_pause_ok:
                raise ValueError(f"the model has pause tokens but {d} has no pause.pt")

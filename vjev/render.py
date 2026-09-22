"""Requests -> token sequences the scorer reads, and batches of them.

One sequence per question; the state (images, then text) opens it:

    Picture 1: <|vision_start|> <|image_pad|> x N <|vision_end|>     (numbered only if several)
    <state text>
    Question: <instructions>
    (A) <option 1>
    (B) <option 2>
    ...
    Answer: (A) (B) ...        <- slots: every option's score is read here (readout="trailing")

A noul question has no options; its slot is the last token of the question. Every option
sits in the same sequence so that options can see each other; the slots are read AFTER
the whole list so that each can see every option (with a causal mask, a slot placed at an
option's own last token could never see the options after it).

N = t*h*w / merge^2 from the image processor's grid: a 512x341 image is a 22x32 patch
grid, 176 image tokens. Token types follow Qwen's processor exactly - image_pad is 1,
everything else 0; the trunk builds its 3-D M-RoPE positions from them, and without them
every image token would get a plain 1-D text position and the model would lose where each
patch sits, silently.
"""
from __future__ import annotations

import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch

from .model import CHOICE, NOUL, SCORE, Batch

QTYPE_IDS = {"choice": CHOICE, "noul": NOUL, "score": SCORE}
READOUTS = ("inline", "trailing")
TEXT, IMAGE = 0, 1          # mm_token_type_ids values
MAX_SIDE = 512              # every training image had its longer side at this


def label(i: int) -> str:
    """A..Z, AA..AZ, ... - up to 255 options."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


@dataclass
class Example:
    input_ids: list[int]
    slots: list[int]
    target: list[float]
    qtype: int
    shard: str
    state_sha1: str
    # where the state (and any image) ends and the question begins; prefix sharing caches
    # the sequence up to here across requests (vjev.prefix)
    state_len: int = 0

    def __len__(self) -> int:
        return len(self.input_ids)


@dataclass
class MMExample(Example):
    """An Example whose sequence opens with one image block per image."""
    mm_types: list[int] = field(default_factory=list)
    image_keys: list[str] = field(default_factory=list)


def options_and_target(row: dict) -> tuple[list[str], list[float]]:
    """Option texts in declaration order, with a target distribution aligned to them
    (uniform placeholders at inference: the renderer expects the field)."""
    qt = row["qtype"]
    if qt == "noul":
        return [], [float(row.get("noul", 0.5))]
    c = row["criteria"]
    if isinstance(c, dict):                       # choice: {key: description}
        keys = list(c)
        p = row.get("probabilities") or {}
        return [c[k] for k in keys], [float(p.get(k, 1 / len(keys))) for k in keys]
    levels = list(c)                              # score: ordered level names
    p = row.get("probabilities") or {}
    return levels, [float(p.get(str(i), 1 / len(levels))) for i in range(len(levels))]


def check_pause(readout: str, n_pause: int, pause_id: int | None, tokenizer) -> None:
    if not n_pause:
        return
    if readout != "trailing":
        raise ValueError("pause tokens sit between the option list and the trailing slots; "
                         "an inline slot comes before them and could never see them")
    if pause_id is None or pause_id < len(tokenizer):
        raise ValueError(f"pause_id {pause_id} must lie past the tokenizer's {len(tokenizer):,} "
                         f"entries (model.pause_token_id), or real text could produce it")


def fit_image(image, max_side: int = MAX_SIDE):
    """Downscale so the longer side is at most `max_side`; never upscale. What the training
    data had: a 512 px image is ~180-260 image tokens; a phone photo at full size is
    thousands - past the sequence budget, and outside anything the model has seen."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    s = max_side / max(w, h)
    return image.resize((max(1, int(w * s)), max(1, int(h * s))))


def linear_patch_embed(visual) -> None:
    """Run the ViT's patch embedding as a matmul instead of a Conv3d.

    The Conv3d's kernel equals its stride equals its whole input (2 x 16 x 16), so it is a
    linear map of each flattened patch. As a convolution, torch 2.9 runs it in bf16 without
    cuDNN: measured on a 4060 Ti, 33-38 s for the ~5,000 patches of 8 images, against
    0.05 s for a whole transformer block. Same weights, same result up to rounding."""
    pe = visual.patch_embed

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        w = pe.proj.weight
        return F.linear(hidden_states.to(w.dtype).view(-1, w[0].numel()), w.view(w.size(0), -1),
                        pe.proj.bias)
    pe.forward = forward


class ImageStore:
    """Images by key. The patch grid is cached per image (it fixes the token count, which
    rendering needs up front); pixels are computed per batch."""

    def __init__(self, image_processor, dirs: Sequence[Path] = ()):
        self.ip = image_processor
        self.merge = image_processor.merge_size
        self.paths: dict[str, Path] = {}
        for d in dirs:
            for p in Path(d).glob("*.jpg"):
                self.paths[p.stem] = p
        self._mem: dict = {}                       # images handed in by a request
        self._grid: dict[str, tuple[int, int, int]] = {}

    def put(self, key: str, image) -> str:
        """Hold a PIL image in memory under `key`; drop() it when the request is done."""
        self._mem[key] = image.convert("RGB")
        return key

    def drop(self, keys: Sequence[str]) -> None:
        for k in keys:
            self._mem.pop(k, None)
            self._grid.pop(k, None)

    def _open(self, key: str):
        from PIL import Image
        img = self._mem.get(key)
        return img if img is not None else Image.open(self.paths[key]).convert("RGB")

    def grid(self, key: str) -> tuple[int, int, int]:
        g = self._grid.get(key)
        if g is None:
            out = self.ip(images=[self._open(key)], return_tensors="pt")
            g = tuple(int(x) for x in out["image_grid_thw"][0])
            self._grid[key] = g
        return g

    def n_tokens(self, key: str) -> int:
        t, h, w = self.grid(key)
        return t * h * w // self.merge ** 2

    def pixels(self, keys: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.ip(images=[self._open(k) for k in keys], return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]


class VisionRenderer:
    """Request row -> MMExample (text-only rows too: they just have no image block)."""

    def __init__(self, tokenizer, store: ImageStore, config, max_len: int = 2560,
                 readout: str = "trailing", n_pause: int = 0, pause_id: int | None = None):
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        check_pause(readout, n_pause, pause_id, tokenizer)
        self.pause_ids = [pause_id] * n_pause
        self.tok, self.store, self.max_len, self.readout = tokenizer, store, max_len, readout
        self.v_start = config.vision_start_token_id
        self.v_pad = config.image_token_id
        self.v_end = config.vision_end_token_id

    def _t(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def render(self, row: dict) -> MMExample | None:
        """`image_keys` (or a single `image_key`) become one image block each, in order;
        `state` text, when present, follows them. None when the sequence is over max_len."""
        opts, target = options_and_target(row)
        qt = QTYPE_IDS[row["qtype"]]
        keys = list(row.get("image_keys") or ([row["image_key"]] if row.get("image_key") else []))

        ids, types = [], []
        # Several images are numbered, in Qwen's own convention ("Picture 1: "), so that a
        # question can say which one it means. A single image stays bare: that is the only
        # layout the training data has.
        numbered = len(keys) > 1 and row.get("label_images", True)
        for i, k in enumerate(keys):
            if numbered:
                lab = self._t(f"{'' if i == 0 else chr(10)}Picture {i + 1}: ")
                ids += lab
                types += [TEXT] * len(lab)
            n = self.store.n_tokens(k)
            ids += [self.v_start] + [self.v_pad] * n + [self.v_end]
            types += [TEXT] + [IMAGE] * n + [TEXT]
        if row.get("state"):
            ids += self._t(row["state"])
        state_len = len(ids)
        ids += self._t(f"\n\nQuestion: {row['instructions']}")

        slots: list[int] = []
        trailing = self.readout == "trailing" and qt != NOUL
        if qt == NOUL:
            ids += self.pause_ids
            slots.append(len(ids) - 1)
        else:
            for i, o in enumerate(opts):
                ids += self._t(f"\n({label(i)}) {o}")
                if not trailing:
                    slots.append(len(ids) - 1)
            if trailing:
                ids += self.pause_ids + self._t("\nAnswer:")
                for i in range(len(opts)):
                    ids += self._t(f" ({label(i)})")
                    slots.append(len(ids) - 1)

        if len(ids) > self.max_len:
            return None
        types += [TEXT] * (len(ids) - len(types))
        return MMExample(ids, slots, target, qt, row.get("shard", "serve"),
                         keys[0] if keys else "", state_len, mm_types=types, image_keys=keys)


def collate(examples: Sequence[Example], pad_id: int) -> Batch:
    B = len(examples)
    S = max(len(e) for e in examples)
    K = max(len(e.slots) for e in examples)

    input_ids = torch.full((B, S), pad_id, dtype=torch.long)
    attn = torch.zeros((B, S), dtype=torch.long)
    slots = torch.zeros((B, K), dtype=torch.long)
    slot_mask = torch.zeros((B, K), dtype=torch.bool)
    target = torch.zeros((B, K), dtype=torch.float)
    qtype = torch.zeros(B, dtype=torch.long)

    for b, e in enumerate(examples):
        n = len(e)
        # Left-pad: slots are absolute indices, and left padding keeps every real token
        # contiguous with the sequence end, which the recurrent layers need.
        input_ids[b, S - n:] = torch.tensor(e.input_ids)
        attn[b, S - n:] = 1
        off = S - n
        k = len(e.slots)
        slots[b, :k] = torch.tensor(e.slots) + off
        slot_mask[b, :k] = True
        target[b, :len(e.target)] = torch.tensor(e.target)
        qtype[b] = e.qtype

    return Batch(input_ids, attn, slots, slot_mask, target, qtype,
                 tuple(e.shard for e in examples))


def collate_mm(examples: Sequence[Example], pad_id: int, store: ImageStore) -> Batch:
    """collate plus the trunk's image inputs. Text rows may share the batch: they get
    all-zero token types and contribute no pixels. Images are concatenated in row order,
    which is the order their image_pad runs appear in the flattened batch."""
    b = collate(examples, pad_id)
    keys = [k for e in examples for k in getattr(e, "image_keys", ())]
    if not keys:
        return b
    B, S = b.input_ids.shape
    types = torch.zeros((B, S), dtype=torch.long)
    for i, e in enumerate(examples):
        mt = getattr(e, "mm_types", None)
        if mt:
            types[i, S - len(e):] = torch.tensor(mt)          # left padding, as collate
    pixel_values, grid = store.pixels(keys)
    b.mm = {"pixel_values": pixel_values, "image_grid_thw": grid, "mm_token_type_ids": types}
    return b


def batches(examples: list[Example], max_tokens: int = 3072, max_rows: int = 8,
            shuffle: bool = False, seed: int = 0, window: int = 512) -> Iterator[list[Example]]:
    """Length-bucketed, token-budget batches: rows of similar length together, so that
    padding a short row out to a long neighbour does not cost 10x."""
    import random
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    order: list[int] = []
    for i in range(0, len(idx), window):
        order += sorted(idx[i:i + window], key=lambda j: len(examples[j]))
    batch: list[Example] = []
    longest = 0
    for j in order:
        e = examples[j]
        cand = max(longest, len(e))
        if batch and (cand * (len(batch) + 1) > max_tokens or len(batch) >= max_rows):
            yield batch
            batch, longest = [], 0
            cand = len(e)
        batch.append(e)
        longest = cand
    if batch:
        yield batch

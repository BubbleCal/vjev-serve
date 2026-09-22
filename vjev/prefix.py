"""Prefix broadcast: encode what a request's questions share once, not once per question.

Every question in a request is its own sequence - image, state, question, options - so
that questions cannot see each other (as the Jev API's are, measured). Scored
naively, the image and the state are re-encoded for every question: 20 questions about one
image process its 176 tokens 20 times, and latency grows linearly with the question count.
The Jev API's does not (1 -> 4,000 questions costs 3x, measured), because it encodes the
state once. This does the same, at inference only:

    1. the longest common token prefix of the sequences is run once, keeping the cache;
    2. the cache is copied across the batch, and each question runs only its own tail.

Causal attention makes this exact rather than approximate: nothing in the prefix can
depend on what follows it. The 8 full-attention layers reuse the prefix's keys and values;
the 24 linear-attention layers reuse its recurrent state and the conv's last few inputs -
transformers' Qwen3.5 layers accept both for a multi-token continuation. That recurrent
state has a fixed size whatever the prefix length, so a 5,000-token state costs no more to
broadcast than a 50-token one.

Tails are RIGHT-padded, unlike everywhere else in this repo. Left padding would put pad
positions between the prefix and the tail, and a linear-attention layer decays its state
across every position, padded or not; after the tail, padding is harmless because nothing
is read there.

Two refinements on top of that:

  a prefix TREE   The sequences of one request share more than one level: the state, then
                  often the question (the same question over several option sets), then
                  often the first options. Each shared run of at least MIN_PREFIX tokens is
                  encoded once, extending a copy of its parent's cache.

  a cache ACROSS  The state's cache is kept (PrefixCache, LRU by bytes), so a second request
  requests        about the same image or document skips its prefix forward altogether.

One constraint shapes both. Keys and values can be cut back to any earlier position; a
recurrent state cannot - it exists only at the position where it was captured. So a cached
prefix can serve only sequences that begin with ALL of it, and the cross-request entry is
captured at a boundary that does not depend on the questions: the end of the state
(Example.state_len), not the request's longest common prefix.

tests/test_prefix.py holds the paths to the same logits.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict, defaultdict
from typing import Sequence

import torch

from .render import Example

MIN_PREFIX = 16       # a state shorter than this is not worth a forward pass of its own
MIN_RUN = 8           # below the state, a branch needs a shared run at least this long ...
MIN_SAVED = 48        # ... that saves this many tokens: (members - 1) x run. A branch costs a
                      # batch-1 forward and a copy of the cache (52 MB of recurrent state).


def _tensors(cache):
    for layer in cache.layers:
        for value in vars(layer).values():
            for t in (value.values() if isinstance(value, dict) else [value]):
                if torch.is_tensor(t):
                    yield t


def broadcast(cache, n: int):
    """A copy of a batch-1 cache, repeated n times along the batch axis (n = 1: a plain copy,
    which every reuse needs because a forward pass appends to the cache it is given).

    transformers' own batch_repeat_interleave covers keys and values only - the linear
    attention layers' conv and recurrent states are left at batch 1 (5.17), and a layer
    that mixes both kinds would half-broadcast. Every tensor a layer holds is repeated here."""
    out = copy.deepcopy(cache)
    if n == 1:
        return out
    rep = lambda t: t.repeat_interleave(n, dim=0) if torch.is_tensor(t) and t.ndim else t
    for layer in out.layers:
        for name, value in vars(layer).items():
            if torch.is_tensor(value):
                setattr(layer, name, rep(value))
            elif isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
                setattr(layer, name, {k: rep(v) for k, v in value.items()})
    return out


class PrefixCache:
    """State prefixes kept across requests, least recently used out first.

    An entry is what the trunk held after the state's last token: keys and values for the
    full-attention layers (32 KB per token on Qwen3.5-4B) and the linear layers' recurrent
    state (52 MB, whatever the length) - 58 MB for one image, 216 MB for a 5,000-token
    document. `start` is the position the next token takes, which after an image is not the
    token count (M-RoPE)."""

    def __init__(self, max_bytes: int = 2 << 30, max_entries: int = 64):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.items: OrderedDict[str, tuple] = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    @staticmethod
    def key(ids: Sequence[int], extra: bytes = b"") -> str:
        """`extra` must identify the images: their placeholder tokens are the same for
        every image of one size, so the token ids alone would confuse two photos."""
        h = hashlib.sha1(extra)
        h.update(torch.tensor(list(ids), dtype=torch.int64).numpy().tobytes())
        return h.hexdigest()

    def get(self, key: str):
        hit = self.items.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        self.items.move_to_end(key)
        return hit[0], hit[1]

    def put(self, key: str, cache, start: int) -> None:
        size = sum(t.numel() * t.element_size() for t in _tensors(cache))
        if size > self.max_bytes:
            return
        if key in self.items:
            self.bytes -= self.items.pop(key)[2]
        self.items[key] = (cache, start, size)
        self.bytes += size
        while self.bytes > self.max_bytes or len(self.items) > self.max_entries:
            self.bytes -= self.items.popitem(last=False)[1][2]


def _lcp(group) -> int:
    """Tokens every tail starts with, leaving each at least one token of its own and every
    slot outside the shared run."""
    first = group[0][1]
    n = min(min(len(ids) - 1, min(slots)) for _, ids, slots in group)
    for _, ids, _ in group[1:]:
        k = 0
        while k < n and ids[k] == first[k]:
            k += 1
        n = k
    return max(n, 0)


def _branches(group) -> list[tuple[list, int]]:
    """Split a group into (subgroup, shared run) pairs; a run of 0 means "no branch worth
    its own forward pass - score these as they are".

    Bucketing by the first token is not enough: every tail of a request opens with the same
    "\\n\\nQuestion:", too short to branch on, and the runs worth sharing (the same question
    over two option sets) lie behind it. So a short common run is walked past - not encoded
    here, each branch encodes it again - and the group is split at the first token where
    its members actually differ."""
    if len(group) < 2:
        return [(group, 0)]
    n = _lcp(group)
    if n >= MIN_RUN and n * (len(group) - 1) >= MIN_SAVED:
        return [(group, n)]
    buckets = defaultdict(list)
    for g in group:
        buckets[g[1][n]].append(g)
    if len(buckets) == 1:             # n was capped by a slot or a length, not by a difference
        return [(group, 0)]
    out = []
    for b in buckets.values():
        out += _branches(b)
    return out


def common_prefix(examples: Sequence[Example]) -> int:
    return _lcp([(i, e.input_ids, e.slots) for i, e in enumerate(examples)]) if examples else 0


class _Run:
    """One score_shared call: the model, where positions stand, and what it has cost."""

    def __init__(self, model, pad_id: int, max_tokens: int, max_rows: int):
        self.model, self.pad_id = model, pad_id
        self.max_tokens, self.max_rows = max_tokens, max_rows
        self.device = next(model.parameters()).device
        self.delta = 0                    # position minus token count, set by an image prefix
        self.copy_budget = 2 << 30        # bytes of cache a tail batch may hold at once
        self.processed = self.segments = 0

    def trunk(self, ids: torch.Tensor, mask: torch.Tensor, past, at: int, **mm):
        S = ids.size(1)
        # With images the multimodal trunk must lay out M-RoPE positions itself; everywhere
        # else they are passed explicitly, because given a mask and a cache it builds them
        # for prefix + tail and then cannot add them to a tail-length input.
        pos = None if mm else (at + self.delta + torch.arange(S, device=self.device)
                               ).view(1, 1, S).expand(3, ids.size(0), S)
        self.processed += int(mask[:, -S:].sum())
        return self.model.trunk(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                position_ids=pos, past_key_values=past, use_cache=True, **mm)

    def extend(self, cache, ids: list[int], at: int, **mm):
        """The cache after `ids`, leaving `cache` itself untouched for its other children."""
        past = broadcast(cache, 1) if cache is not None else None
        tok = torch.tensor([ids])
        out = self.trunk(tok, torch.ones((1, at + len(ids)), dtype=torch.long), past, at, **mm)
        self.segments += 1
        if mm:
            inner = self.model.trunk.get_base_model() if hasattr(self.model.trunk, "get_base_model") \
                else self.model.trunk
            self.delta = int(inner.rope_deltas.reshape(-1)[0])
        return out.past_key_values

    def descend(self, group, cache, at: int, rows: list) -> None:
        """group: (example index, remaining ids, slots relative to them), all sharing `cache`."""
        leaves = []
        for sub, n in _branches(group):
            if n:
                deeper = self.extend(cache, sub[0][1][:n], at)
                self.descend([(i, ids[n:], [s - n for s in slots]) for i, ids, slots in sub],
                             deeper, at + n, rows)
            else:
                leaves += sub
        if leaves:
            self.tails(leaves, cache, at, rows)

    def tails(self, group, cache, at: int, rows: list) -> None:
        group = sorted(group, key=lambda g: len(g[1]))
        # every row of a batch gets its own copy of the cache, and the recurrent state alone
        # is 52 MB a row on the 4B model - 64 rows would be 3.3 GB before any activations
        size = sum(t.numel() * t.element_size() for t in _tensors(cache)) if cache is not None else 0
        cap = min(self.max_rows, max(1, self.copy_budget // size)) if size else self.max_rows
        batch: list = []
        for g in group + [None]:
            full = batch and (g is None or len(batch) >= cap
                              or len(g[1]) * (len(batch) + 1) > self.max_tokens)
            if full:
                B, S = len(batch), max(len(ids) for _, ids, _ in batch)
                tok = torch.full((B, S), self.pad_id, dtype=torch.long)
                mask = torch.zeros((B, S), dtype=torch.long)
                for b, (_, ids, _) in enumerate(batch):
                    tok[b, :len(ids)] = torch.tensor(ids)        # right-padded: see module doc
                    mask[b, :len(ids)] = 1
                past = broadcast(cache, B) if cache is not None else None
                whole = torch.cat([torch.ones((B, at), dtype=torch.long), mask], dim=1)
                h = self.trunk(tok, whole, past, at).last_hidden_state
                for b, (i, _, slots) in enumerate(batch):
                    idx = torch.tensor(slots, device=self.device)
                    rows[i] = self.model.head(h[b, idx].float()).squeeze(-1)
                batch = []
            if g is not None:
                batch.append(g)


@torch.no_grad()
def score_shared(model, examples: Sequence[Example], pad_id: int, mm: dict | None = None,
                 max_tokens: int = 4096, max_rows: int = 64, store: PrefixCache | None = None,
                 key_extra: bytes = b"") -> tuple[list[torch.Tensor], dict]:
    """Logits per example (its own slots only), sharing every common run of tokens.

    `mm` is the image input of ONE row (pixel_values, image_grid_thw) as vision.collate_mm
    builds it; every example must carry the same images and the same state, which is what a
    request is. `store` keeps the state's cache for later calls; `key_extra` must identify
    the images (PrefixCache.key)."""
    run = _Run(model, pad_id, max_tokens, max_rows)
    first = examples[0]
    B = first.state_len
    same_state = all(e.state_len == B and e.input_ids[:B] == first.input_ids[:B] for e in examples)
    if mm and not (same_state and B >= sum(getattr(first, "mm_types", ()))):
        raise ValueError("examples with images must share one state that contains them all")

    cache, at, hit = None, 0, None
    if same_state and B >= MIN_PREFIX:
        key = PrefixCache.key(first.input_ids[:B], key_extra)
        got = store.get(key) if store is not None else None
        hit = got is not None
        if hit:
            cache, start = got
            run.delta = start - B
        else:
            kw = {}
            if mm:
                kw = {"pixel_values": mm["pixel_values"].to(run.device),
                      "image_grid_thw": mm["image_grid_thw"].to(run.device),
                      "mm_token_type_ids": torch.tensor([first.mm_types[:B]], device=run.device)}
            cache = run.extend(None, first.input_ids[:B], 0, **kw)
            if store is not None:
                store.put(key, cache, B + run.delta)     # never written to again: every use copies
        at = B

    rows: list = [None] * len(examples)
    run.descend([(i, e.input_ids[at:], [s - at for s in e.slots]) for i, e in enumerate(examples)],
                cache, at, rows)
    return rows, {"prefix": at, "processed": run.processed, "segments": run.segments,
                  "cache": None if hit is None else ("hit" if hit else "miss")}

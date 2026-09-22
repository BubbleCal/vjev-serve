"""Prefix broadcast (vjev.prefix) must give the logits the plain path gives. CPU, fp32, a
shrunken random Qwen3.5 with both layer kinds - and a prefix long enough to cross the
gated delta rule's 64-token chunk boundary, where a wrong initial state would show.

  python tests/test_prefix.py          (VJEV_MODEL=<hub id or local dir> to pick the tokenizer)

With flash-linear-attention installed the linear layers call triton kernels, which need a
GPU; the linear-attention half is skipped there and runs wherever fla is absent.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402

from vjev.model import ListwiseScorer                                 # noqa: E402
from vjev.prefix import PrefixCache, score_shared                     # noqa: E402
from vjev.render import ImageStore, VisionRenderer, collate           # noqa: E402

BASE = os.environ.get("VJEV_MODEL", "yah01/vjev-vision-pilot")     # only its tokenizer and config are read
STATE = ("Ticket 4471. The customer upgraded to Pro on Monday and was charged $49, but the "
         "account still shows the Free tier and the export button is greyed out. They logged "
         "out and back in, cleared the cache, and tried a second browser. Support replied once "
         "with a canned answer about billing cycles, which did not address the missing "
         "entitlement. They are now asking for either the upgrade to be applied or a refund, "
         "and mention a deadline on Friday for a report that needs the export.")
ROWS = [
    {"qtype": "choice", "instructions": "Which team owns this?", "shard": "t",
     "criteria": {"a": "billing", "b": "provisioning and entitlements", "c": "frontend bug"},
     "probabilities": {"a": 0.3, "b": 0.6, "c": 0.1}},
    {"qtype": "score", "instructions": "How urgent is it?", "shard": "t",
     "criteria": ["low", "medium", "high"], "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}},
    {"qtype": "noul", "instructions": "The customer was charged.", "shard": "t", "noul": 0.9},
    {"qtype": "noul", "instructions": "The customer was charged twice for the same upgrade and "
     "has already received a refund.", "shard": "t", "noul": 0.1},
]


def renderer(tok):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(BASE)
    return VisionRenderer(tok, None, cfg, readout="trailing")


def tiny(layer_types):
    from transformers import AutoConfig, AutoModelForCausalLM
    t = copy.deepcopy(AutoConfig.from_pretrained(BASE).text_config)
    t.num_hidden_layers, t.layer_types = len(layer_types), layer_types
    t.hidden_size, t.intermediate_size, t.head_dim = 64, 128, 16
    t.num_attention_heads, t.num_key_value_heads = 4, 2
    t.linear_num_key_heads, t.linear_num_value_heads = 2, 4
    t.linear_key_head_dim = t.linear_value_head_dim = 16
    torch.manual_seed(0)
    trunk = AutoModelForCausalLM.from_config(t).model.float().eval()
    trunk.config.use_cache = False
    return ListwiseScorer(trunk, 64).eval()


# one long question over three option sets, and four statements that open alike: two
# branches below the state for the tree to find
LONG_Q = ("Considering only what the customer wrote and not what support replied, which of the "
          "following teams should take ownership of this ticket from here on?")
LONG_S = ("Setting aside the missing export entirely, and judging only by the charge that appears "
          "on the statement, it is fair to say that the customer ")
TREE = ROWS + [
    {"qtype": "choice", "instructions": LONG_Q, "shard": "t",
     "criteria": {"a": "billing", "b": "provisioning and entitlements", "c": "frontend bug"},
     "probabilities": {"a": 0.3, "b": 0.6, "c": 0.1}},
    {"qtype": "choice", "instructions": LONG_Q, "shard": "t",
     "criteria": {"x": "legal", "y": "billing", "z": "provisioning and entitlements", "w": "sales"},
     "probabilities": {"x": 0.1, "y": 0.3, "z": 0.5, "w": 0.1}},
    {"qtype": "choice", "instructions": LONG_Q, "shard": "t",
     "criteria": {"p": "sales", "q": "frontend bug"}, "probabilities": {"p": 0.5, "q": 0.5}},
] + [{"qtype": "noul", "instructions": LONG_S + tail, "shard": "t", "noul": 0.5}
     for tail in ("was charged once.", "was charged twice.", "is owed a refund.", "was never charged.")]


def setup(layer_types, rows=ROWS, state=STATE):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    rend = renderer(tok)
    ex = [rend.render(dict(r, state=state)) for r in rows]
    model = tiny(layer_types)
    b = collate(ex, pad_id=0)
    with torch.no_grad():
        ref = model(b.input_ids, b.attention_mask, b.slots, b.slot_mask)
    return model, ex, ref, int(b.attention_mask.sum())


def same(rows, ref, ex, what):
    for i, e in enumerate(ex):
        want = ref[i, :len(e.slots)]
        assert torch.allclose(rows[i], want, atol=2e-4, rtol=1e-4), \
            f"{what}: row {i} differs by {float((rows[i] - want).abs().max()):.2e}"


MIXED = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]


def layer_sets():
    if importlib.util.find_spec("fla") is not None:
        print("      (linear-attention layers skipped: fla is installed and needs a GPU)")
        return [["full_attention", "full_attention"]]
    return [["full_attention", "full_attention"], MIXED]


def test_sharing_the_state_gives_the_plain_logits():
    for lt in layer_sets():
        model, ex, ref, plain = setup(lt)
        assert ex[0].state_len > 64, "the state must cross a 64-token chunk boundary"
        rows, st = score_shared(model, ex, pad_id=0, max_rows=3)          # 3: two tail batches
        same(rows, ref, ex, lt)
        assert st["prefix"] == ex[0].state_len and st["processed"] < plain


def test_the_tree_shares_below_the_state_too():
    for lt in layer_sets():
        model, ex, ref, _ = setup(lt, TREE)
        rows, st = score_shared(model, ex, pad_id=0)
        same(rows, ref, ex, lt)
        assert st["segments"] >= 3, f"expected the state plus two deeper shared runs, got {st}"
        flat = sum(len(e) - e.state_len for e in ex) + ex[0].state_len
        assert st["processed"] < flat, "the deeper levels saved nothing"


def test_a_second_request_reuses_the_cached_state():
    for lt in layer_sets():
        model, ex, ref, _ = setup(lt)
        store = PrefixCache()
        first, a = score_shared(model, ex[:2], pad_id=0, store=store)
        again, b = score_shared(model, ex, pad_id=0, store=store)         # new questions, same state
        third, c = score_shared(model, ex, pad_id=0, store=store)
        assert (a["cache"], b["cache"], c["cache"]) == ("miss", "hit", "hit")
        assert b["processed"] == sum(len(e) - e.state_len for e in ex), "the state was re-encoded"
        same(again, ref, ex, lt)
        same(third, ref, ex, lt)      # the stored entry must not have been written to by its users


def test_the_cache_key_separates_images_and_evicts_the_oldest():
    ids = list(range(40))
    assert PrefixCache.key(ids, b"photo-a") != PrefixCache.key(ids, b"photo-b")
    assert PrefixCache.key(ids, b"x") == PrefixCache.key(ids, b"x")
    model, ex, _, _ = setup(["full_attention"])
    store = PrefixCache(max_entries=2)
    for tag in (b"a", b"b", b"c"):
        score_shared(model, ex, pad_id=0, store=store, key_extra=tag)
    assert len(store.items) == 2 and store.bytes == sum(v[2] for v in store.items.values())
    assert score_shared(model, ex, pad_id=0, store=store, key_extra=b"a")[1]["cache"] == "miss"
    assert score_shared(model, ex, pad_id=0, store=store, key_extra=b"c")[1]["cache"] == "hit"


def test_nothing_to_share_still_scores_correctly():
    for rows in (ROWS[:1], ROWS):                         # one question; a state too short to cache
        model, ex, ref, _ = setup(["full_attention", "full_attention"], rows,
                                  STATE if len(rows) == 1 else "Hi.")
        got, st = score_shared(model, ex, pad_id=0)
        same(got, ref, ex, f"{len(rows)} rows")


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:                                   # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)

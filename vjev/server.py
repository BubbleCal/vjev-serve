"""The vjev inference service: a Jev-shaped API, plus images, and a web console.

  vjev-serve --model yah01/vjev-vision                       # from the Hub (or a local dir)
  vjev-serve --stub                                          # API + UI, no model, no GPU

  /                 web UI: drop an image, write questions, see the distributions
  POST /v1/systemone   the API
  GET  /v1/models      what is loaded
  GET  /v1/presets     option sets a choice without criteria can resolve to
  GET  /health

The request is Jev's:

  {"model": "vjev-latest",
   "state": "free text" | {...} | [...] | [content blocks],
   "questions": {"route": {"type": "choice", "instructions": "...",
                           "criteria": {"billing": "Payments.", "bug": "Defects."}},
                 "urgent": {"type": "noul", "instructions": "Needs attention today."},
                 "sev":    {"type": "score", "instructions": "How severe?",
                            "criteria": ["minor", "moderate", "critical"]}}}

Images are the one extension over Jev. Pass them as content blocks, the way the multimodal
APIs do:

  "state": [{"type": "text", "text": "Frame from the warehouse camera."},
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": "..."}}]
  (an {"type": "image", "url": "data:image/jpeg;base64,..."} block is accepted too)

Answers keep Jev's fields, including `confidence`, which is exactly
(p_max - 1/n) / (1 - 1/n) - a rescaling of p_max carrying no extra information. It is here
for drop-in compatibility; `probabilities` is the real output.

Serving shape: one sequence per question. The state (images and text) is encoded once per
request and kept for later requests about the same state (vjev.prefix); each question runs
only its own tail. `usage` reports both the Jev-style token count and what was processed.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MAX_CHOICES = 255           # Jev's limit; ours is not lower
MAX_QUESTIONS = 256
MAX_IMAGES = 8
QTYPES = ("noul", "choice", "score", "multilabel")


class ApiError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(str(detail))
        self.status, self.detail = status, detail


def err_usage(message: str, status: int = 400) -> ApiError:
    return ApiError(status, {"error_type": "api_usage_error", "message": message})


# ---------------------------------------------------------------- request parsing

@dataclass
class Parsed:
    text: str
    images: list                     # PIL images, in the order they appeared
    questions: dict[str, dict]
    share_prefix: bool = True        # "share_prefix": false re-encodes the state per question
    label_images: bool = True        # "label_images": false leaves several images unnumbered


def _image_from_block(b: dict):
    from PIL import Image
    src = b.get("source") or {}
    data = src.get("data")
    if not data and isinstance(b.get("url"), str):
        m = re.match(r"data:(image/[\w.+-]+);base64,(.*)$", b["url"], re.S)
        if not m:
            raise err_usage("image url must be a data: URI with base64 image data")
        data = m.group(2)
    if not isinstance(data, str):
        raise err_usage("image block needs source.data (base64) or a data: url")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise err_usage("image data is not valid base64")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise err_usage("image data could not be decoded as an image")


def parse_request(body: dict) -> Parsed:
    if not isinstance(body, dict):
        raise ApiError(422, [{"loc": ["body"], "msg": "expected a JSON object"}])
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise ApiError(422, [{"loc": ["body", "questions"],
                              "msg": "questions must be a non-empty object"}])
    if len(qs) > MAX_QUESTIONS:
        raise err_usage(f"Too many questions. Must have at most {MAX_QUESTIONS}.")

    state, text, images = body.get("state", ""), "", []
    if isinstance(state, list) and state and all(
            isinstance(x, dict) and x.get("type") in ("text", "image") for x in state):
        parts = []
        for b in state:
            if b["type"] == "image":
                images.append(_image_from_block(b))
            elif isinstance(b.get("text"), str):
                parts.append(b["text"])
        text = "\n".join(parts)
    elif isinstance(state, str):
        text = state
    elif state is None:
        text = ""
    else:
        text = json.dumps(state, ensure_ascii=False)      # Jev serialises objects/arrays
    if len(images) > MAX_IMAGES:
        raise err_usage(f"Too many images. Must have at most {MAX_IMAGES}.")

    clean: dict[str, dict] = {}
    for name, q in qs.items():
        if not isinstance(q, dict):
            raise ApiError(422, [{"loc": ["body", "questions", name], "msg": "expected an object"}])
        t = q.get("type")
        if t == "bounding_box":
            raise err_usage("Bounding-box questions are not implemented in vjev.")
        if t not in QTYPES:
            raise ApiError(422, [{"type": "union_tag_invalid", "loc": ["body", "questions", name],
                                  "msg": f"Input tag {t!r} found using 'type' does not match any "
                                         f"of the expected tags: {', '.join(QTYPES)}"}])
        instr = q.get("instructions")
        if not isinstance(instr, str) or not instr.strip():
            raise ApiError(422, [{"loc": ["body", "questions", name, "instructions"],
                                  "msg": "instructions must be a non-empty string"}])
        crit = q.get("criteria")
        extra = {}
        if t == "choice" and (crit is None or crit == {} or isinstance(crit, str)):
            # no options given: resolve_presets() picks them. "@name" pins the preset.
            if isinstance(crit, str) and crit.lstrip("@") not in PRESETS:
                raise err_usage(f"Unknown preset {crit!r}. Available: {', '.join(PRESETS)}.")
            if q.get("multiple") not in (None, True, False):
                raise ApiError(422, [{"loc": ["body", "questions", name, "multiple"],
                                      "msg": "multiple must be true or false"}])
            extra = {"preset": crit.lstrip("@") if isinstance(crit, str) else None,
                     "multiple": q.get("multiple"), "auto": True}
            crit = None
        elif t == "choice":
            if not isinstance(crit, dict) or len(crit) < 2:
                raise ApiError(422, [{"loc": ["body", "questions", name, "criteria"],
                                      "msg": "choice needs an object of at least 2 options"}])
            if len(crit) > MAX_CHOICES:
                raise err_usage(f"Too many choices. Must have at most {MAX_CHOICES} choices.")
            crit = {str(k): (str(v) if v is not None else str(k)) for k, v in crit.items()}
        elif t == "multilabel":
            # labels scored independently of one another: resolve_presets() fans them out
            if isinstance(crit, str):
                if crit.lstrip("@") not in PRESETS:
                    raise err_usage(f"Unknown preset {crit!r}. Available: {', '.join(PRESETS)}.")
                crit = dict(PRESETS[crit.lstrip("@")]["labels"])
            elif isinstance(crit, list) and crit:
                crit = {str(x): str(x) for x in crit}
            elif isinstance(crit, dict) and crit:
                crit = {str(k): (str(v) if v is not None else str(k)) for k, v in crit.items()}
            else:
                raise ApiError(422, [{"loc": ["body", "questions", name, "criteria"],
                                      "msg": "multilabel needs labels: a list, an object of "
                                             "label -> description, or \"@preset\""}])
            if len(crit) > MAX_CHOICES:
                raise err_usage(f"Too many labels. Must have at most {MAX_CHOICES}.")
            levels = q.get("levels")
            if levels is not None and (not isinstance(levels, list) or len(levels) < 2):
                raise ApiError(422, [{"loc": ["body", "questions", name, "levels"],
                                      "msg": "levels must be an ordered list of at least 2"}])
            extra = {"levels": [str(x) for x in levels] if levels else None}
        elif t == "score":
            if not isinstance(crit, list) or len(crit) < 2:
                raise ApiError(422, [{"loc": ["body", "questions", name, "criteria"],
                                      "msg": "score needs an ordered list of at least 2 levels"}])
            if len(crit) > MAX_CHOICES:
                raise err_usage(f"Too many levels. Must have at most {MAX_CHOICES}.")
            crit = [str(x) for x in crit]
        else:
            crit = None
        clean[str(name)] = {"type": t, "instructions": instr.strip(), "criteria": crit, **extra}
    return Parsed(text, images, clean, share_prefix=body.get("share_prefix") is not False,
                  label_images=body.get("label_images") is not False)


# ---------------------------------------------------------------- presets
#
# A choice that arrives without options is answered in two passes, both by this same
# model and neither generating a token:
#   1. route   - the user's QUESTION becomes the state, and the model picks which preset it
#                asks for (or "none"), and whether several answers can hold at once.
#   2. rewrite - several: one noul per label, scored independently ("which colours" has no
#                single answer, and a softmax over colours could not say "red and blue");
#                one: an ordinary choice over the preset's labels.
# The route never sees the state or the image: the same question must resolve to the same
# options whatever it is asked about, or answers stop being comparable across inputs.
# Measured on the pilot checkpoint: 18/18 test questions routed right (8 presets + none,
# English and Chinese), 13/13 colours found with no false positive on synthetic images.

PRESETS: dict[str, dict] = {k: v for k, v in json.loads(
    (HERE / "presets.json").read_text(encoding="utf-8")).items()
    if not k.startswith("_")}
ROUTE_MIN = 0.6            # below this the question is not treated as a preset's
ROUTE_MEMO = 4096          # routed questions remembered per engine
SEP = "\x1f"               # joins a question's name to a label; cannot occur in JSON keys we accept
ROUTE_Q = "What kind of answer is the user's question asking for?"
MULTI_Q = "The user's question can have several correct answers at the same time."
NONE = "None of the above - it asks for something else."
STATEMENT = 'Question: "{question}" - "{label}" is one of the correct answers.'
GRADED = "How true is this? {statement}"


def statement(question: str, label: str) -> str:
    """What one label of a multilabel question is judged as. A description written as a
    sentence is judged as written; a bare word is put into the question's frame. Measured on
    one photo: sentences as written separate true from false by 0.70 (0.63 in the frame),
    bare words by 0.35 as written and 0.54 in the frame."""
    text = label.strip()
    if text.endswith((".", "。", "!", "?")) and len(text.split()) >= 3:
        return text
    return STATEMENT.format(question=question, label=text)


def fan_out(name: str, question: str, labels: dict, levels: list | None) -> dict:
    """One independent question per label: a noul, or with `levels` a score on that scale.
    Nothing here competes - that is the point. A score's or a choice's options share one
    softmax, so rating five attributes in one of them makes their scores sum to 1."""
    out = {}
    for key, text in labels.items():
        st = statement(question, text)
        out[f"{name}{SEP}{key}"] = (
            {"type": "score", "instructions": GRADED.format(statement=st), "criteria": list(levels)}
            if levels else {"type": "noul", "instructions": st, "criteria": None})
    return out


def resolve_presets(engine, p: Parsed) -> tuple[Parsed, dict, int]:
    """-> (request with every preset question rewritten, how each was resolved, router tokens)."""
    resolved, questions, spent = {}, {}, 0
    for name, q in p.questions.items():
        if SEP in name:
            raise err_usage("question names may not contain control characters")
        if q["type"] == "multilabel":
            questions.update(fan_out(name, q["instructions"], q["criteria"], q["levels"]))
            continue
        if not q.get("auto"):
            questions[name] = q
            continue
        route = {"route": {"type": "choice", "instructions": ROUTE_Q,
                           "criteria": {**{k: v["about"] for k, v in PRESETS.items()}, "none": NONE}},
                 "multi": {"type": "noul", "instructions": MULTI_Q, "criteria": None}}
        memo = engine.__dict__.setdefault("route_memo", OrderedDict())
        if q["preset"] and q["multiple"] is not None:      # nothing left for the router to decide
            probs, p_multi = {}, None
        elif q["instructions"] in memo:
            # the router reads the question and nothing else, so its answer is a pure function
            # of that string: the second "what colours are in this picture?" costs no forward
            memo.move_to_end(q["instructions"])
            probs, p_multi = memo[q["instructions"]]
        else:
            ans, use = engine.score(Parsed(f"A user asks the following question.\n\nQuestion: "
                                           f"{q['instructions']}", [], route))
            spent += use.get("tokens_processed", 0)
            probs, p_multi = ans["route"]["probabilities"], ans["multi"]["noul"]
            memo[q["instructions"]] = (probs, p_multi)
            while len(memo) > ROUTE_MEMO:
                memo.popitem(last=False)
        preset = q["preset"] or max(probs, key=probs.get)
        info = {"preset": preset, "pinned": bool(q["preset"]),
                "router": {k: v for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]},
                "multiple_p": p_multi}
        if not q["preset"] and (preset == "none" or probs[preset] < ROUTE_MIN):
            raise ApiError(422, [{"loc": ["body", "questions", name, "criteria"],
                                  "msg": "no criteria were given and the question matched no "
                                         "preset option set; pass criteria, or name one as "
                                         f"\"@preset\". Available: {', '.join(PRESETS)}",
                                  "router": info["router"]}])
        spec = PRESETS[preset]
        # the single/multiple reading is weak (it spans 0.33-0.63 on clear-cut questions), so
        # near 0.5 the preset's own default decides; the caller's word always wins
        multiple = q["multiple"] if q["multiple"] is not None else \
            spec["multiple"] if abs(p_multi - 0.5) < 0.05 else p_multi >= 0.5
        info["multiple"] = multiple
        resolved[name] = info
        if multiple:
            questions.update(fan_out(name, q["instructions"], spec["labels"], None))
        else:
            questions[name] = {"type": "choice", "instructions": q["instructions"],
                               "criteria": dict(spec["labels"])}
    return Parsed(p.text, p.images, questions, p.share_prefix, p.label_images), resolved, spent


def fold_presets(answers: dict, resolved: dict) -> dict:
    """Undo resolve_presets' fan-out: a question's nouls become one multilabel answer."""
    out: dict[str, dict] = {}
    for name, a in answers.items():
        base, _, key = name.partition(SEP)
        if key and a["type"] == "noul":
            m = out.setdefault(base, {"type": "multilabel", "labels": [], "probabilities": {}})
            m["probabilities"][key] = a["noul"]
            if a["noul"] >= 0.5:
                m["labels"].append(key)
        elif key:
            # graded: the expected level, scaled to 0-1 so labels compare across scales
            m = out.setdefault(base, {"type": "multilabel", "labels": [], "scores": {},
                                      "legend": a["legend"], "distributions": {}})
            m["scores"][key] = round(a["score"] / (len(a["legend"]) - 1), 4)
            m["distributions"][key] = a["probabilities"]
            if m["scores"][key] >= 0.5:
                m["labels"].append(key)
        else:
            out[name] = a
    for name, info in resolved.items():
        out[name]["resolved"] = info
    return out


def confidence(probs: list[float]) -> float:
    """Jev's field: chance-corrected p_max."""
    n = len(probs)
    return 0.0 if n < 2 else max(0.0, (max(probs) - 1 / n) / (1 - 1 / n))


def answer_of(q: dict, probs: list[float]) -> dict:
    t = q["type"]
    if t == "noul":
        return {"type": "noul", "noul": round(probs[0], 4)}
    if t == "choice":
        keys = list(q["criteria"])
        p = {k: round(v, 4) for k, v in zip(keys, probs)}
        return {"type": "choice", "choice": max(p, key=p.get),
                "confidence": round(confidence(probs), 4), "probabilities": p}
    levels = q["criteria"]
    return {"type": "score", "score": round(sum(i * v for i, v in enumerate(probs)), 4),
            "confidence": round(confidence(probs), 4),
            "legend": {str(i): l for i, l in enumerate(levels)},
            "probabilities": {str(i): round(v, 4) for i, v in enumerate(probs)}}


# ---------------------------------------------------------------- engines

class StubEngine:
    """Deterministic pseudo-answers so the API and UI can be exercised without a GPU."""
    model_id, vision, meta = "vjev-stub", True, {"stub": True}

    def score(self, p: Parsed) -> tuple[dict, dict]:
        answers, tokens = {}, 0
        for name, q in p.questions.items():
            n = 1 if q["type"] == "noul" else len(q["criteria"])
            seed = hashlib.sha1(f"{p.text}|{len(p.images)}|{q['instructions']}".encode()).digest()
            raw = [(seed[i % len(seed)] + 7 * i) % 97 / 12 for i in range(n)]
            if q["type"] == "noul":
                probs = [1 / (1 + pow(2.718281828, -(raw[0] - 4)))]
            else:
                m = max(raw)
                ex = [pow(2.718281828, r - m) for r in raw]
                probs = [e / sum(ex) for e in ex]
            answers[name] = answer_of(q, probs)
            tokens += 15 + 4 * n
        return answers, {"input_tokens": len(p.text) // 4 + tokens + 256 * len(p.images),
                         "output_tokens": sum(1 if q["type"] == "noul" else len(q["criteria"])
                                              for q in p.questions.values()),
                         "tokens_processed": 0}


class ModelEngine:
    """The real thing: one sequence per question, the shared state encoded once."""

    def __init__(self, repo: str, max_tokens: int, max_image_side: int = 512,
                 prefix_cache_mb: int = 2048, device: str | None = None):
        import torch
        from .loading import load
        from .model import pause_token_id
        from .prefix import PrefixCache
        from .render import ImageStore, VisionRenderer

        self.torch = torch
        self.lock = threading.Lock()
        self.max_tokens, self.max_image_side = max_tokens, max_image_side
        self.prefix_cache = PrefixCache(max_bytes=prefix_cache_mb << 20)
        t0 = time.time()
        L = load(repo, device)
        self.model, self.meta, self.device = L.model, L.meta, L.device
        self.vision = True
        self.tok = L.processor.tokenizer
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.store = ImageStore(L.processor.image_processor)
        self.render = VisionRenderer(self.tok, self.store, L.config, max_len=max_tokens,
                                     readout=self.meta.get("readout", "trailing"),
                                     n_pause=int(self.meta.get("pause", 0)),
                                     pause_id=pause_token_id(L.config)).render
        self.model_id = L.model_id
        print(f"loaded {self.model_id} on {self.device} in {time.time() - t0:.0f}s: "
              f"readout={self.meta.get('readout')} step={self.meta.get('step')}", flush=True)

    def score(self, p: Parsed) -> tuple[dict, dict]:
        from .model import NOUL
        from .prefix import score_shared
        from .render import batches, collate_mm, fit_image
        keys, img_tokens = [], 0
        state_tokens = len(self.tok(p.text, add_special_tokens=False).input_ids) if p.text else 0
        try:
            with self.lock:
                sizes, digest = [], hashlib.sha1()
                for i, img in enumerate(p.images):
                    small = fit_image(img, self.max_image_side)
                    sizes.append({"received": list(img.size), "used": list(small.size)})
                    # the prefix cache's key: image placeholder tokens do not tell photos apart
                    digest.update(repr(small.size).encode() + small.tobytes())
                    keys.append(self.store.put(f"req{id(p)}_{i}", small))
                img_tokens = sum(self.store.n_tokens(k) for k in keys)   # before drop()
                rows, order = [], []
                for name, q in p.questions.items():
                    crit = q["criteria"]
                    row = {"qtype": q["type"], "instructions": q["instructions"],
                           "state": p.text, "image_keys": keys, "shard": "serve",
                           "label_images": p.label_images}
                    if q["type"] == "choice":
                        row["criteria"] = crit
                        row["probabilities"] = {k: 1 / len(crit) for k in crit}
                    elif q["type"] == "score":
                        row["criteria"] = crit
                        row["probabilities"] = {str(i): 1 / len(crit) for i in range(len(crit))}
                    else:
                        row["noul"] = 0.5
                    e = self.render(row)
                    if e is None:
                        raise ApiError(400, {"error_type": "max_tokens_exceeded",
                                             "message": f"question {name!r} with the state exceeds "
                                                        f"{self.max_tokens} tokens"})
                    rows.append(e)
                    order.append((name, q))
                idx = {id(e): i for i, e in enumerate(rows)}
                probs: list = [None] * len(rows)
                processed, st = 0, None
                # the image and the state once - kept for the next request about them - and
                # below that every run of tokens the questions share (vjev.prefix)
                if p.share_prefix:
                    mm1 = collate_mm(rows[:1], self.pad_id, self.store).mm if keys else None
                    logits, st = score_shared(self.model, rows, self.pad_id, mm=mm1,
                                              max_tokens=self.max_tokens, max_rows=16,
                                              store=self.prefix_cache, key_extra=digest.digest())
                    processed = st["processed"]
                    probs = [[float(self.torch.sigmoid(lg[0]))] if e.qtype == NOUL
                             else self.torch.softmax(lg, -1).tolist() for lg, e in zip(logits, rows)]
                with self.torch.no_grad():
                    for group in batches(rows if st is None else [], max_tokens=self.max_tokens,
                                         shuffle=False):
                        b = collate_mm(group, self.pad_id, self.store).to(self.device)
                        processed += int(b.attention_mask.sum())
                        lg = self.model(b.input_ids, b.attention_mask, b.slots, b.slot_mask,
                                        **(b.mm or {}))
                        for j, e in enumerate(group):
                            k = len(e.slots)
                            probs[idx[id(e)]] = ([float(self.torch.sigmoid(lg[j, 0]))]
                                                 if e.qtype == NOUL else
                                                 self.torch.softmax(lg[j, :k].float(), -1).tolist())
        finally:
            self.store.drop(keys)
        answers = {name: answer_of(q, probs[i]) for i, (name, q) in enumerate(order)}
        return answers, {"input_tokens": state_tokens + img_tokens
                         + sum(self.question_tokens(q) for _, q in order),
                         "output_tokens": sum(1 if q["type"] == "noul" else len(q["criteria"])
                                              for _, q in order),
                         "tokens_processed": processed,
                         **({"prefix_tokens_shared": st["prefix"], "prefix_cache": st["cache"],
                             "shared_runs": st["segments"]} if st else {}),
                         # what each image was scaled to before the model saw it
                         **({"images": sizes} if sizes else {})}

    def question_tokens(self, q: dict) -> int:
        """Jev bills a question by its own text, the shared state once per request; score
        criteria are an ordered list and choice criteria a mapping, hence the two arms."""
        n = len(self.tok(q["instructions"], add_special_tokens=False).input_ids)
        c = q["criteria"]
        if isinstance(c, dict):
            n += sum(len(self.tok(f"({k}) {v}", add_special_tokens=False).input_ids)
                     for k, v in c.items())
        elif isinstance(c, list):
            n += sum(len(self.tok(str(x), add_special_tokens=False).input_ids) for x in c)
        return n


# ---------------------------------------------------------------- HTTP

def make_handler(engine, page: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "vjev"

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", flush=True)

        def _send(self, body: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # never cached: a browser holding yesterday's page against today's API read a
            # "multilabel" answer as a score and crashed on its missing legend
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type, authorization")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(json.dumps(obj).encode(), "application/json; charset=utf-8", status)

        def do_OPTIONS(self):
            self._send(b"", "text/plain", 204)

        def do_GET(self):
            path = self.path.split("?")[0].strip("/")
            if path == "":
                self._send(page.encode(), "text/html; charset=utf-8")
            elif path == "health":
                self._json({"ok": True, "model": engine.model_id, "vision": engine.vision})
            elif path == "v1/presets":
                self._json({"presets": {k: {"about": v["about"], "multiple": v["multiple"],
                                            "labels": v["labels"]} for k, v in PRESETS.items()}})
            elif path == "v1/models":
                self._json({"models": [{"id": engine.model_id, "vision": engine.vision,
                                        **engine.meta}]})
            else:
                self._json({"detail": "not found"}, 404)

        def do_POST(self):
            if self.path.split("?")[0].strip("/") != "v1/systemone":
                return self._json({"detail": "not found"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            t0 = time.perf_counter()
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                return self._json({"detail": [{"loc": ["body"], "msg": f"invalid JSON: {e}"}]}, 422)
            try:
                parsed = parse_request(body)
                parsed, resolved, routed = resolve_presets(engine, parsed)
                answers, usage = engine.score(parsed)
                answers = fold_presets(answers, resolved)
                if resolved:
                    usage["router_tokens_processed"] = routed
            except ApiError as e:
                return self._json({"detail": e.detail}, e.status)
            except Exception as e:                                    # noqa: BLE001
                print(f"!! {type(e).__name__}: {e}", flush=True)
                return self._json({"detail": {"error_type": "internal_error",
                                              "message": f"{type(e).__name__}: {e}"}}, 500)
            self._json({"model": engine.model_id, "answers": answers, "usage": usage,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yah01/vjev-vision",
                    help="a Hugging Face model id, or a local directory with the same files")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best available)")
    ap.add_argument("--stub", action="store_true", help="serve fake answers, load no model")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--prefix-cache-mb", type=int, default=2048,
                    help="memory for state prefixes kept across requests (~58 MB per image, "
                         "~216 MB per 5,000-token document); 0 keeps none")
    ap.add_argument("--max-image-side", type=int, default=512,
                    help="images are downscaled to this before scoring: the training resolution")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    engine = StubEngine() if a.stub else ModelEngine(a.model, a.max_tokens, a.max_image_side,
                                                     a.prefix_cache_mb, a.device)
    page = (HERE / "page.html").read_text(encoding="utf-8")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(engine, page))
    print(f"serving {engine.model_id} on http://{a.host}:{a.port}  "
          f"(/, POST /v1/systemone, /v1/models, /health)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

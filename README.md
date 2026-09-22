# vjev

Typed decisions from text and images, in one forward pass. Give the model a *state* (text,
images, or both) and *questions* — is this statement true? which of these? how much on this
scale? — and it returns calibrated probabilities for every option. Nothing is generated.

It is a re-creation of the [Jev](https://typesafe.ai) API's shape on an open base
(Qwen3.5-4B), with images added. This repository is the inference side: the model code, a
Jev-compatible HTTP API, and a web console. The weights are on the Hub:
[`yah01/vjev-vision-pilot`](https://huggingface.co/yah01/vjev-vision-pilot).

## Run it

```bash
pip install git+https://github.com/BubbleCal/vjev-serve
vjev-serve --model yah01/vjev-vision-pilot           # downloads ~9 GB the first time
```

Then open <http://localhost:8800>: drop an image, write questions, see the distributions.
CUDA (bf16, ~9 GB), Apple silicon (fp16, ~10 GB) and CPU (fp32, slow) all work; the device
is picked automatically (`--device` to force one). `--stub` serves the API and the console
with fake answers, for trying the interface without a model.

## The API

`POST /v1/systemone`:

```json
{"state": [{"type": "text", "text": "Frame from the warehouse camera, 12:40."},
           {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}],
 "questions": {
   "person": {"type": "noul",   "instructions": "There is a person in this image."},
   "where":  {"type": "choice", "instructions": "Where is the forklift?",
              "criteria": {"left": "left half", "right": "right half", "none": "no forklift"}},
   "busy":   {"type": "score",  "instructions": "How cluttered is the scene?",
              "criteria": ["empty", "sparse", "busy", "crowded"]},
   "tags":   {"type": "multilabel", "instructions": "Which of these describe the scene?",
              "criteria": ["indoors", "daylight", "crowded", "hazard"]}}}
```

```json
{"answers": {
   "person": {"type": "noul", "noul": 0.93},
   "where":  {"type": "choice", "choice": "left", "confidence": 0.71,
              "probabilities": {"left": 0.80, "right": 0.14, "none": 0.06}},
   "busy":   {"type": "score", "score": 1.4, "legend": {"0": "empty", "1": "sparse", "2": "busy", "3": "crowded"},
              "probabilities": {"0": 0.12, "1": 0.45, "2": 0.34, "3": 0.09}},
   "tags":   {"type": "multilabel", "labels": ["indoors", "daylight"],
              "probabilities": {"indoors": 0.91, "daylight": 0.77, "crowded": 0.22, "hazard": 0.08}}},
 "usage": {"input_tokens": 231, "tokens_processed": 312, "prefix_cache": "miss"},
 "latency_ms": 640}
```

- `state` may also be a plain string, or a JSON object/array (serialised, as Jev does).
  Images: up to 8, as content blocks (base64 `source`, or a `data:` URL); each is resized to a
  512 px longer side, the training resolution. Several images are numbered — a question can
  say "the person in picture 1".
- **noul** — one probability that the statement holds. **choice** — one softmax over the
  options: they compete. **score** — the levels of one ordered scale; `score` is the expected
  level. **multilabel** — each label scored on its own, nothing competes (internally one
  noul per label, or a score per label if you pass `levels`). To rate several independent
  things, use multilabel or several nouls, not one choice.
- A choice sent **without `criteria`** is matched to a preset option set (colours, objects,
  sentiment, emotion, scene, weather, topic, language — `GET /v1/presets`): the model itself
  reads the question and picks the preset and whether several answers can hold. `"@colors"`
  pins one.
- Questions never see each other; options within a question do, and are read in the order
  given (reordering them moves probabilities by ~0.03 TVD on average).
- `confidence` is Jev's `(p_max - 1/n) / (1 - 1/n)`, kept for compatibility; `probabilities`
  is the output.

Other routes: `GET /v1/models`, `GET /v1/presets`, `GET /health`.

## In Python

```python
from vjev import load
from vjev.server import ModelEngine   # or drive the pieces yourself, as server.py does

L = load("yah01/vjev-vision-pilot")   # -> model, processor, config, meta, device
```

`vjev/render.py` turns a question into the sequence the model reads, `vjev/model.py` is the
listwise head over Qwen3.5's trunk, `vjev/prefix.py` encodes what a request's questions share
once and keeps the state across requests.

## How it works, briefly

Every option of a question lives in one sequence, and every option's score is read at a
slot *after* the whole list (`Answer: (A) (B) …`), so options can see each other — a
pointwise scorer, one option per pass, cannot reproduce the option interactions the Jev API
shows. The base's vocabulary projection is replaced by one shared linear head; there is no
decoding. A request's shared prefix (the images and the state) is encoded once and its
cache broadcast to every question, then kept for the next request about the same state.

## Limits of the current checkpoint

It is a pilot (300 vision steps). On held-out COCO geometry questions: choice accuracy 0.67,
ECE 0.09; VQAv2: 0.65 / 0.03. Yes/no presence judgements lean towards "yes" on adversarial
absent objects (13.8% false positives on POPE-adversarial). Trained on single images.
Details on the model card.

## License

Apache-2.0. The base model, Qwen3.5-4B, is Apache-2.0.

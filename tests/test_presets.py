"""Preset resolution in vjev/server.py, against a scripted engine: no model loaded.

  python tests/test_presets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vjev import server as js                                          # noqa: E402


class Scripted:
    """Routes every question to `preset` with probability `p`; says `multi` to 'several?'.
    Any other question gets 0.9 on a noul and mass on its first option."""

    def __init__(self, preset="colors", p=0.95, multi=0.63):
        self.preset, self.p, self.multi, self.calls = preset, p, multi, []

    def score(self, parsed):
        self.calls.append(parsed)
        out = {}
        for name, q in parsed.questions.items():
            if q["instructions"] == js.ROUTE_Q:
                keys = list(q["criteria"])
                rest = (1 - self.p) / (len(keys) - 1)
                out[name] = js.answer_of(q, [self.p if k == self.preset else rest for k in keys])
            elif q["instructions"] == js.MULTI_Q:
                out[name] = js.answer_of(q, [self.multi])
            elif q["type"] == "noul":
                out[name] = js.answer_of(q, [0.9 if '"red"' in q["instructions"] else 0.1])
            else:
                n = len(q["criteria"])
                out[name] = js.answer_of(q, [0.7] + [0.3 / (n - 1)] * (n - 1))
        return out, {"tokens_processed": 10}


def ask(engine, question, **extra):
    p = js.parse_request({"state": "a state", "questions": {"q": {"type": "choice", "instructions": question, **extra}}})
    p2, resolved, _ = js.resolve_presets(engine, p)
    answers, _ = engine.score(p2)
    return p2, js.fold_presets(answers, resolved)


def test_several_answers_become_one_noul_per_label_and_fold_back():
    eng = Scripted("colors", multi=0.63)
    p2, ans = ask(eng, "What colors are in this picture?")
    assert len(p2.questions) == len(js.PRESETS["colors"]["labels"])
    assert all(q["type"] == "noul" and "What colors are in this picture?" in q["instructions"]
               for q in p2.questions.values()), "the user's question must survive the rewrite"
    a = ans["q"]
    assert a["type"] == "multilabel" and a["labels"] == ["red"]
    assert set(a["probabilities"]) == set(js.PRESETS["colors"]["labels"])
    assert a["resolved"]["preset"] == "colors" and a["resolved"]["multiple"] is True


def test_the_router_never_sees_the_state_or_the_images():
    """Same question -> same options, whatever it is asked about."""
    eng = Scripted("colors")
    ask(eng, "What colors are in this picture?")
    routed = eng.calls[0]
    assert "a state" not in routed.text and routed.images == []
    assert "What colors are in this picture?" in routed.text


def test_one_answer_becomes_a_plain_choice():
    _, ans = ask(Scripted("weather", multi=0.30), "What's the weather like?")
    a = ans["q"]
    assert a["type"] == "choice" and set(a["probabilities"]) == set(js.PRESETS["weather"]["labels"])
    assert a["resolved"]["multiple"] is False


def test_a_weak_several_reading_falls_back_to_the_presets_default():
    assert ask(Scripted("colors", multi=0.48), "colors?")[1]["q"]["type"] == "multilabel"   # default: several
    assert ask(Scripted("weather", multi=0.52), "weather?")[1]["q"]["type"] == "choice"     # default: one
    assert ask(Scripted("colors", multi=0.52), "c?", multiple=False)[1]["q"]["type"] == "choice"  # caller wins


def test_no_preset_is_an_error_that_shows_the_router():
    for eng in (Scripted("none", p=0.8), Scripted("topic", p=0.4)):          # none, and too unsure
        try:
            ask(eng, "Which team should handle this ticket?")
        except js.ApiError as e:
            assert e.status == 422 and "router" in e.detail[0]
            continue
        raise AssertionError("resolved a question that matched no preset")


def test_pinned_preset_with_pinned_mode_skips_the_router():
    eng = Scripted("colors")
    _, ans = ask(eng, "How does the customer feel?", criteria="@sentiment", multiple=False)
    assert len(eng.calls) == 1, "the router ran with nothing to decide"
    assert ans["q"]["resolved"] == {"preset": "sentiment", "pinned": True, "router": {},
                                    "multiple_p": None, "multiple": False}
    try:
        js.parse_request({"questions": {"q": {"type": "choice", "instructions": "x", "criteria": "@nope"}}})
    except js.ApiError as e:
        assert e.status == 400
    else:
        raise AssertionError("an unknown preset was accepted")


def test_a_question_is_routed_once():
    eng = Scripted("colors")
    _, first = ask(eng, "What colors are in this picture?")
    n = len(eng.calls)                                   # router + scoring
    _, again = ask(eng, "What colors are in this picture?")
    assert len(eng.calls) == n + 1, "the router ran again for a question it had already read"
    assert again["q"]["resolved"] == first["q"]["resolved"]
    ask(eng, "What colors are on the car?")              # a different string is a different question
    assert len(eng.calls) == n + 3


def scored(engine, q):
    p = js.parse_request({"state": "a state", "questions": {"q": q}})
    p2, resolved, _ = js.resolve_presets(engine, p)
    answers, _ = engine.score(p2)
    return p2, js.fold_presets(answers, resolved)


def test_multilabel_scores_each_label_on_its_own():
    eng = Scripted()
    p2, ans = scored(eng, {"type": "multilabel", "instructions": "Which describe the cat?",
                           "criteria": ["red", "fluffy", "asleep"]})
    assert len(eng.calls) == 1, "explicit labels need no router"
    assert [q["type"] for q in p2.questions.values()] == ["noul"] * 3
    a = ans["q"]
    assert a["type"] == "multilabel" and a["labels"] == ["red"]
    assert a["probabilities"] == {"red": 0.9, "fluffy": 0.1, "asleep": 0.1}
    assert sum(a["probabilities"].values()) != 1.0, "independent labels have no reason to sum to 1"


def test_a_sentence_is_judged_as_written_and_a_word_is_framed():
    p2, _ = scored(Scripted(), {"type": "multilabel", "instructions": "Which describe the cat?",
                                "criteria": {"asleep": "The cat is asleep.", "fluffy": "fluffy"}})
    text = {k.split(js.SEP)[1]: q["instructions"] for k, q in p2.questions.items()}
    assert text["asleep"] == "The cat is asleep."
    assert "Which describe the cat?" in text["fluffy"] and '"fluffy"' in text["fluffy"]


def test_multilabel_with_levels_grades_each_label_on_that_scale():
    levels = ["not at all", "slightly", "very"]
    p2, ans = scored(Scripted(), {"type": "multilabel", "instructions": "Describe the cat.",
                                  "criteria": ["fluffy", "asleep"], "levels": levels})
    assert all(q["type"] == "score" and q["criteria"] == levels for q in p2.questions.values())
    a = ans["q"]
    # Scripted puts 0.7 on the first level: expected level 0.15 + 2 * 0.15 = 0.45, over 2
    assert a["scores"] == {"fluffy": 0.225, "asleep": 0.225} and a["labels"] == []
    assert set(a["distributions"]["fluffy"]) == {"0", "1", "2"} and a["legend"]["2"] == "very"


def test_multilabel_takes_a_preset_and_refuses_nonsense():
    p2, _ = scored(Scripted(), {"type": "multilabel", "instructions": "colors?", "criteria": "@colors"})
    assert len(p2.questions) == len(js.PRESETS["colors"]["labels"])
    for bad in ({"criteria": []}, {"criteria": None}, {"criteria": ["a"], "levels": ["only one"]}):
        try:
            js.parse_request({"questions": {"q": {"type": "multilabel", "instructions": "x", **bad}}})
        except js.ApiError as e:
            assert e.status == 422
            continue
        raise AssertionError(f"accepted {bad}")


def test_explicit_criteria_are_left_alone():
    eng = Scripted()
    p2, ans = ask(eng, "Which team?", criteria={"billing": "payments", "auth": "login"})
    assert len(eng.calls) == 1 and "resolved" not in ans["q"]
    assert list(p2.questions["q"]["criteria"]) == ["billing", "auth"]


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

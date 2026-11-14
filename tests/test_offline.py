from bench.analyze import evaluate_policy, fit_temperature, flip_rate, pick_thresholds, reliability, temper
from jev.decide import build_prompt, route, softmax
from jev.generate import parse
from train.lora import examples
import random


def test_softmax_matches_known_values():
    p = softmax([8.2, 5.5, 4.8])
    assert [round(x, 2) for x in p] == [0.91, 0.06, 0.03]


def test_prompt_labels_follow_choice_order():
    msg, labels = build_prompt("t", "x", {"billing": "b", "other": "o"})
    assert labels == {"A": "billing", "B": "other"} and "A = b" in msg


def test_route_policy():
    sure = {"choice": "billing", "probabilities": {"billing": .91, "other": .09}}
    tied = {"choice": "billing", "probabilities": {"billing": .46, "other": .44}}
    esc = {"choice": "other", "probabilities": {"billing": .05, "other": .95}}
    assert (route(sure), route(tied), route(esc)) == ("auto", "review", "review")


def test_parse_takes_earliest_whole_word():
    ch = {"billing_fees": "", "other": "", "card_issues": ""}
    assert parse("Billing fees. Not another card_issues case.", ch) == "billing_fees"
    assert parse("It is another thing", ch) is None  # 'another' is not 'other'


def _rows(n, conf, acc):  # n rows, top prob `conf`, a fraction `acc` of them correct
    return [{"label": "a" if i < n * acc else "b", "probs": {"a": conf, "b": 1 - conf}} for i in range(n)]


def test_temperature_fixes_overconfidence():
    rows = _rows(100, 0.99, 0.7)  # says 99%, is right 70%
    T = fit_temperature(rows)
    assert T > 1 and reliability(rows, T)[1] < reliability(rows)[1]
    assert abs(sum(temper({"a": .9, "b": .1}, 2).values()) - 1) < 1e-9


def test_threshold_and_flip():
    rows = _rows(50, 0.95, 1.0) + _rows(50, 0.55, 0.5)
    p, m, e = pick_thresholds(rows, 0.95)
    assert e["coverage"] == 0.5 and e["auto_acc"] == 1.0 and (p > 0.55 or m > 0.1)
    assert evaluate_policy(rows, 0.5, 0)["coverage"] == 1.0
    assert flip_rate(["a", "b"], ["a", "a"]) == 0.5


def test_training_examples_match_inference_labels():
    row = {"task": "t", "text": "x", "choices": {"billing": "b", "tech": "t", "other": "o"}, "label": "tech"}
    seen = set()
    for seed in range(20):
        (msg, letter), = examples([row], random.Random(seed))
        lines = dict(l.split(" = ") for l in msg.split("Allowed labels:\n")[1].split("\n\n")[0].split("\n"))
        assert lines[letter] == "t"  # the target letter really points at the 'tech' description
        seen.add(letter)
    assert len(seen) > 1  # order is shuffled, so the letter moves


if __name__ == "__main__":
    test_softmax_matches_known_values(); test_prompt_labels_follow_choice_order(); test_route_policy()
    test_parse_takes_earliest_whole_word(); test_temperature_fixes_overconfidence(); test_threshold_and_flip(); test_training_examples_match_inference_labels(); print("ok")

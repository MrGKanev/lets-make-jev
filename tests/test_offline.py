from jev.decide import build_prompt, route, softmax


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


if __name__ == "__main__":
    test_softmax_matches_known_values(); test_prompt_labels_follow_choice_order(); test_route_policy(); print("ok")

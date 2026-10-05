"""Fixed-choice scoring: one /v1/score call returns a probability per choice."""
import math

import requests
from transformers import AutoTokenizer

OTHER = "other"  # escape route so the softmax never forces a wrong answer
LABELS = "ABCDEFGHIJ"


def build_prompt(task, text, choices):
    """choices: {key: description}. Returns (user_message, {label: key})."""
    labels = dict(zip(LABELS, choices))
    lines = "\n".join(f"{l} = {choices[k]}" for l, k in labels.items())
    msg = f"{task}\n\nInput:\n{text}\n\nAllowed labels:\n{lines}\n\nReturn only the label."
    return msg, labels


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


class Jev:
    def __init__(self, model, base_url="http://127.0.0.1:30000"):
        self.model, self.url = model, base_url
        self.tok = AutoTokenizer.from_pretrained(model)
        self._ids = {}

    def _label_ids(self, labels):
        # The answer position is the first assistant token, so tokenize the bare label
        # and reject anything that is not exactly one token.
        out = []
        for l in labels:
            if l not in self._ids:
                ids = self.tok.encode(l, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"{l!r} is not a single token: {ids}")
                self._ids[l] = ids[0]
            out.append(self._ids[l])
        return out

    def decide(self, task, text, choices):
        """Returns {"choice": key, "probabilities": {key: p}}."""
        msg, labels = build_prompt(task, text, choices)
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )
        r = requests.post(
            f"{self.url}/v1/score",
            json={
                "model": self.model,
                "query": prompt,
                "items": [""],
                "label_token_ids": self._label_ids(labels),
                "apply_softmax": True,
            },
            timeout=120,
        )
        r.raise_for_status()
        probs = dict(zip(labels.values(), r.json()["scores"][0]))
        return {"choice": max(probs, key=probs.get), "probabilities": probs}


def route(result, min_p=0.80, min_margin=0.20):
    """Policy lives in code: automate only when the winner is clear, else review."""
    top, second = sorted(result["probabilities"].values(), reverse=True)[:2]
    clear = top >= min_p and top - second >= min_margin and result["choice"] != OTHER
    return "auto" if clear else "review"

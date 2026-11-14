"""Fixed-choice scoring: one /v1/score call returns a probability per choice."""
import math
import json
import hashlib
from pathlib import Path

from jev import http

OTHER = "other"  # escape route so the softmax never forces a wrong answer
LABELS = "ABCDEFGHIJ"


def build_prompt(task, text, choices):
    """choices: {key: description}. Returns (user_message, {label: key})."""
    if not 2 <= len(choices) <= len(LABELS):
        raise ValueError("choices must contain between 2 and 10 categories")
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
    def __init__(self, model, base_url="http://127.0.0.1:30000", tok=None, policy=None):
        self.model, self.url = model, base_url
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model)
        self.policy = json.loads(Path(policy).read_text()) if policy else None
        if self.policy:
            if self.policy["model"] != model:
                raise ValueError("Policy model does not match")
            if self.policy["prompt_version"] != PROMPT_VERSION:
                raise ValueError("Unsupported policy prompt version")
            if not math.isfinite(self.policy["temperature"]) or self.policy["temperature"] <= 0:
                raise ValueError("Policy temperature must be finite and positive")
            if not all(0 <= self.policy[k] <= 1 for k in ("min_p", "min_margin")):
                raise ValueError("Policy thresholds must be in [0, 1]")
        self.tok = tok
        self._ids = {}
        self.backend = "score"  # flips to "generate" if the server has no usable /v1/score

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
        """Returns {"choice": key, "probabilities": {key: p}}. Dict order = label order."""
        msg, labels = build_prompt(task, text, choices)
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )
        ids = self._label_ids(labels)
        scores = self._scores(prompt, ids)
        validate_scores(scores, len(labels))
        probs = dict(zip(labels.values(), scores))
        return self._result(task, choices, probs)

    def decide_many(self, task, texts, choices):
        """Same task and choices for every text: one /v1/score call, the shared prefix is the query,
        each item is `text + rest of the template`. Falls back to one call per text."""
        texts = list(texts)
        if not texts:
            return []
        mark = "\x00TEXT\x00"
        msg, labels = build_prompt(task, mark, choices)
        head, tail = self.tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True).split(mark)
        if self.backend == "score":
            r = http.post(
                f"{self.url}/v1/score",
                json={"model": self.model, "query": head, "items": [t + tail for t in texts],
                      "label_token_ids": self._label_ids(labels), "apply_softmax": True},
                timeout=300,
            )
            if r.ok:
                out = []
                for sc in score_rows(r, len(texts), len(labels)):
                    probs = dict(zip(labels.values(), sc))
                    out.append(self._result(task, choices, probs))
                return out
            if r.status_code not in (400, 404, 405, 422, 501):
                r.raise_for_status()
            self.backend = "generate"
        return [self.decide(task, t, choices) for t in texts]

    def _result(self, task, choices, probs):
        if list(probs) != list(choices):
            raise http.ResponseError("Probability categories do not match choices")
        validate_scores(list(probs.values()), len(choices))
        raw = dict(probs)
        if self.policy:
            if self.policy["prompt_fingerprint"] != prompt_fingerprint(task, choices):
                raise ValueError("Policy task, choices or prompt version does not match")
            temperature = self.policy["temperature"]
            weights = {k: max(v, 1e-12) ** (1 / temperature) for k, v in probs.items()}
            probs = {k: v / sum(weights.values()) for k, v in weights.items()}
        result = {"choice": max(probs, key=probs.get), "probabilities": probs,
                  "raw_probabilities": raw, "calibrated": self.policy is not None}
        p = self.policy or {"min_p": .8, "min_margin": .2}
        result["route"], result["review_reasons"] = policy_decision(result, p["min_p"], p["min_margin"])
        return result

    def _scores(self, prompt, ids):
        if self.backend == "score":
            r = http.post(
                f"{self.url}/v1/score",
                json={"model": self.model, "query": prompt, "items": [""],
                      "label_token_ids": ids, "apply_softmax": True},
                timeout=120,
            )
            if r.ok:
                return score_rows(r, 1, len(ids))[0]
            if r.status_code not in (400, 404, 405, 422, 501):
                r.raise_for_status()
            # An empty item is the shakiest assumption; fall back instead of dying.
            self.backend = "generate"
        # Fallback: one-token native /generate with logprobs for just our label tokens,
        # softmaxed over those tokens client-side (same restricted softmax).
        r = http.post(
            f"{self.url}/generate",
            json={"text": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                  "return_logprob": True, "token_ids_logprob": ids},
            timeout=120,
        )
        r.raise_for_status()
        try:
            got = r.json()["meta_info"]["output_token_ids_logprobs"][0]
            by_id = {t[1]: t[0] for t in got}
            values = [by_id[i] for i in ids]
            if len(by_id) != len(got) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("Missing, duplicate or non-finite label logprobs")
            return softmax(values)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise http.ResponseError("Invalid native logprob response") from exc



def route(result, min_p=0.80, min_margin=0.20):
    """Policy lives in code: automate only when the winner is clear, else review."""
    return policy_decision(result, min_p, min_margin)[0]


PROMPT_VERSION = "1"


def prompt_fingerprint(task, choices):
    payload = json.dumps([PROMPT_VERSION, task, list(choices.items())], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def policy_decision(result, min_p=.8, min_margin=.2):
    top, second = sorted(result["probabilities"].values(), reverse=True)[:2]
    reasons = []
    if result["choice"] == OTHER:
        reasons.append("other")
    if top < min_p:
        reasons.append("low_confidence")
    if top - second < min_margin:
        reasons.append("small_margin")
    return ("review" if reasons else "auto"), reasons


def validate_scores(values, n):
    if not isinstance(values, list) or len(values) != n:
        raise http.ResponseError("Wrong number of label probabilities")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise http.ResponseError("Probabilities must be finite numbers in [0, 1]")
    if not math.isclose(sum(values), 1, abs_tol=1e-3):
        raise http.ResponseError("Probabilities must sum to 1")


def score_rows(response, n_rows, n_labels):
    try:
        rows = response.json()["scores"]
    except (ValueError, KeyError, TypeError) as exc:
        raise http.ResponseError("Invalid score response") from exc
    if not isinstance(rows, list) or len(rows) != n_rows:
        raise http.ResponseError("Wrong number of scored items")
    for values in rows:
        validate_scores(values, n_labels)
    return rows

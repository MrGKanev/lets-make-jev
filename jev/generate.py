"""Baseline lane: ordinary chat generation, then parse the answer out of the text."""
import requests

from jev.decide import build_prompt


def generate(model, task, text, choices, base_url="http://127.0.0.1:30000", max_tokens=32):
    msg, _ = build_prompt(task, text, choices)
    msg = msg.replace("Return only the label.", "Answer with the category name and a short reason.")
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": msg}],
              "max_tokens": max_tokens, "temperature": 0},
        timeout=120,
    )
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    head = out[:100].lower()
    # ponytail: first-match parse on the head of the text; fine for distinct keys, upgrade if keys overlap
    hits = [k for k in choices if k.replace("_", " ") in head or k in head]
    return {"choice": hits[0] if hits else None, "text": out}

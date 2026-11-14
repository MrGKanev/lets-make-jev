"""Baseline lane: ordinary chat generation, then parse the answer out of the text."""
import json
import re

from jev import http


def generate(model, task, text, choices, base_url="http://127.0.0.1:30000", max_tokens=32):
    # A normal chat user names the categories (no A/B/C trick) and lets the model write.
    lines = "\n".join(f"{k} = {d}" for k, d in choices.items())
    msg = f"{task}\n\nInput:\n{text}\n\nAllowed categories:\n{lines}\n\nAnswer with the category name and a short reason."
    r = http.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": msg}],
              "max_tokens": max_tokens, "temperature": 0},
        timeout=120,
    )
    r.raise_for_status()
    out = http.chat_content(r)
    return {"choice": parse(out, choices), "text": out}


def parse(out, choices):
    """Earliest whole-word mention of a choice key in the first 100 chars (a later mention is usually the reason)."""
    head, best = out[:100].lower(), (None, 10**9)
    for k in choices:
        m = re.search(r"\b" + k.lower().replace("_", "[ _]") + r"\b", head)
        if m and m.start() < best[1]:
            best = (k, m.start())
    return best[0]


def generate_structured(model, task, text, choices, base_url="http://127.0.0.1:30000"):
    """Third lane: constrained JSON decoding ({"choice": <one of the keys>}), no parsing guesswork."""
    lines = "\n".join(f"{k} = {d}" for k, d in choices.items())
    msg = f"{task}\n\nInput:\n{text}\n\nAllowed categories:\n{lines}\n\nReply with JSON."
    schema = {"type": "object", "properties": {"choice": {"enum": list(choices)}}, "required": ["choice"]}
    r = http.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": 48, "temperature": 0,
              "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}}},
        timeout=120,
    )
    r.raise_for_status()
    out = http.chat_content(r)
    try:
        choice = json.loads(out)["choice"]
        return {"choice": choice if choice in choices else None, "text": out}
    except (ValueError, KeyError, TypeError):
        return {"choice": None, "text": out}


def generate_label(model, task, text, choices, base_url="http://127.0.0.1:30000"):
    """Matched baseline: identical scoring prompt, one generated token, no reason."""
    from jev.decide import build_prompt
    msg, labels = build_prompt(task, text, choices)
    r = http.post(f"{base_url}/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": msg}],
        "max_tokens": 1, "temperature": 0}, timeout=120)
    r.raise_for_status()
    text = http.chat_content(r)
    return {"choice": labels.get(text.strip()), "text": text}

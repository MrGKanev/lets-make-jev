"""Whole pipeline (bench.full + figures) against a fake SGLang. Proves the plumbing, says nothing about models.

    python -m tests.smoke_fake      # no GPU, no network, no transformers; writes to a temp dir
"""
import json
import math
import re
import tempfile
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import bench.full as full
from article import figures


def overlap_scores(prompt, sep):
    """Fake 'model': score each option by words shared between the input and its description."""
    head, options = prompt.rsplit("\n\nAllowed", 1)
    text = head.split("Input:\n", 1)[1].lower().split()
    opts = re.findall(rf"^(\w+) {sep} (.*)$", options, re.M)
    return [(k, sum(w in d.lower() for w in text) + (0.5 if k == "other" else 0)) for k, d in opts]


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({})

    def do_POST(self):
        if self.path == "/flush_cache":
            return self._send({"ok": True})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/score":
            out = []
            for item in body["items"]:  # real server: item continues the shared query
                s = [v * 3 for _, v in overlap_scores(body["query"] + item, "=")]
                e = [math.exp(x - max(s)) for x in s]
                out.append([v / sum(e) for v in e])
            self._send({"scores": out})
        elif self.path == "/generate":  # native endpoint used by the fallback
            s = [v * 3 for _, v in overlap_scores(body["text"], "=")]
            e = [math.exp(x - max(s)) for x in s]
            ids = body["token_ids_logprob"]
            self._send({"meta_info": {"output_token_ids_logprobs": [[[math.log(v / sum(e)), i, None] for v, i in zip(e, ids)]]}})
        else:
            s = overlap_scores(body["messages"][0]["content"], "=")
            if "response_format" in body:
                return self._send({"choices": [{"message": {"content": json.dumps({"choice": max(s, key=lambda t: t[1])[0]})}}]})
            if body.get("max_tokens") == 1:
                return self._send({"choices": [{"message": {"content": max(s, key=lambda t: t[1])[0]}}]})
            self._send({"choices": [{"message": {"content": max(s, key=lambda t: t[1])[0].replace("_", " ") + " because"}}]})


class Tok:  # one label letter == one token, like the real thing
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return msgs[0]["content"]

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    full.Jev = lambda model, base, tok=None: __import__("jev.decide", fromlist=["Jev"]).Jev(model, base, tok=Tok())
    a = Namespace(seed=0, n=40, reps=2, perms=2, target_acc=0.8, batch=16, cold_cache=True, min_auto=10, policy_criterion="observed")
    R = full.run_model("fake/model", url, a, "news")
    out = Path(tempfile.mkdtemp())
    (out / "fake.json").write_text(json.dumps(R))
    figures.IMG, figures.HERE = out / "img", out
    figures.main(out)
    from bench.report import write_report
    write_report(R, out / "diagnostics.md")
    assert (out / "diagnostics.risk.png").exists()
    assert R["solo"]["label"][0]["requests"]
    assert R["cold_cache"]["label"]["n"] == 40
    assert R["challenge"]["by_tag"]["injection"]
    assert R["diagnostics"]["per_class"]["other"]["support"] > 0
    assert R["backend"] == "score" and not R["sanity"]["warn"], R["sanity"]
    assert R["sanity"]["max_abs_diff"] < 1e-9
    assert R["batched"][0]["agree_with_single"] == 1.0 and R["solo"]["structured"], R["batched"]
    assert (out / "tables.md").exists() and len(list((out / "img").glob("*.png"))) == 6
    print("smoke ok", out, "| fake acc:", round(R["solo"]["score"][0]["acc"], 2))

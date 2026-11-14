"""Live demo: two columns, same cases, score lane vs generate lane running side by side.

    python3 -m ui.serve --model Qwen/Qwen2.5-0.5B-Instruct --launch           # starts SGLang too
    python3 -m ui.serve --model Qwen/Qwen2.5-0.5B-Instruct --url http://127.0.0.1:30000
Open http://localhost:8000
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jev.decide import Jev, policy_decision
from jev.generate import generate

ROOT = Path(__file__).resolve().parent.parent


def make_handler(jev, url, cases, thresholds):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path == "/cases":
                return self._json([{"text": c["text"], "label": c["label"]} for c in cases])
            b = (ROOT / "ui/index.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_POST(self):  # /score or /generate, body {"i": case index}
            c = cases[json.loads(self.rfile.read(int(self.headers["Content-Length"])))["i"]]
            t = time.perf_counter()
            if self.path == "/score":
                r = jev.decide(c["task"], c["text"], c["choices"])
                decision, reasons = (r["route"], r["review_reasons"]) if jev.policy else policy_decision(r, *thresholds)
                out = {"choice": r["choice"], "probs": r["probabilities"], "route": decision,
                       "review_reasons": reasons,
                       "calibrated": r["calibrated"]}
            else:
                r = generate(jev.model, c["task"], c["text"], c["choices"], url)
                out = {"choice": r["choice"], "text": r["text"]}
            out["ms"] = 1000 * (time.perf_counter() - t)
            out["ok"] = out["choice"] == c["label"]
            self._json(out)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", help="calibration policy exported by bench.full")
    ap.add_argument("--domain", default="support")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--launch", action="store_true", help="start SGLang for --model first")
    ap.add_argument("--n", type=int, default=50, help="how many test cases to show")
    ap.add_argument("--min-p", type=float, default=0.80)
    ap.add_argument("--min-margin", type=float, default=0.20)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    proc = None
    if a.launch:
        from bench.full import start_server
        proc = start_server(a.model, 30000, ROOT / "bench/results/ui.server.log")
    cases = [json.loads(l) for l in open(ROOT / f"data/{a.domain}_test.jsonl")][:a.n]
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(Jev(a.model, a.url, policy=a.policy), a.url, cases, (a.min_p, a.min_margin)))
    print(f"open http://localhost:{a.port}", flush=True)
    try:
        srv.serve_forever()
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()

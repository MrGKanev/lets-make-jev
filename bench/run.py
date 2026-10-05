"""Run score and generate lanes concurrently on one server. Usage: python -m bench.run MODEL data/x.jsonl"""
import json
import statistics as st
import sys
import threading
import time

from jev.decide import Jev
from jev.generate import generate


def lane(name, fn, cases, out):
    start.wait()
    t0 = time.perf_counter()
    for c in cases:
        t = time.perf_counter()
        pick = fn(c)
        out.append({"lat": time.perf_counter() - t, "ok": pick == c["label"]})
    out.append({"total": time.perf_counter() - t0})


def summary(rows):
    r = [x for x in rows if "lat" in x]
    lat = sorted(x["lat"] for x in r)
    return {"n": len(r), "acc": sum(x["ok"] for x in r) / len(r),
            "p50_ms": 1000 * st.median(lat), "p95_ms": 1000 * lat[int(.95 * (len(lat) - 1))],
            "total_s": rows[-1]["total"]}


if __name__ == "__main__":
    model, path = sys.argv[1:3]
    rows = [json.loads(l) for l in open(path)]  # {task, text, choices, label}
    jev = Jev(model)
    start = threading.Barrier(2)
    res = {"score": [], "generate": []}
    fns = {
        "score": lambda c: jev.decide(c["task"], c["text"], c["choices"])["choice"],
        "generate": lambda c: generate(model, c["task"], c["text"], c["choices"])["choice"],
    }
    ts = [threading.Thread(target=lane, args=(k, fns[k], rows, res[k])) for k in fns]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(json.dumps({k: summary(v) for k, v in res.items()}, indent=2))

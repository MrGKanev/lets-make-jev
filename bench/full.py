"""One command, end to end: server -> warm-up -> solo/concurrent lanes -> order bias -> calibration -> JSON.

    python -m bench.full --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-4B-Instruct-2507
    python -m bench.full --models Qwen/Qwen2.5-0.5B-Instruct --smoke      # quick API check
    python -m bench.full --models M --url http://127.0.0.1:30000          # server already running
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import platform
import random
import statistics as st
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

import requests

from bench import analyze as an
from bench.experiment import experiment
from data.validate import audit
from jev.decide import Jev, prompt_fingerprint, PROMPT_VERSION
from jev.generate import generate, generate_structured, generate_label

ROOT = Path(__file__).resolve().parent.parent


def load(path, n=None):
    rows = [json.loads(l) for l in open(ROOT / path)]
    for i, row in enumerate(rows):
        row.setdefault("id", f"{Path(path).stem}:{i}")
    return random.Random(0).sample(rows, n) if n else rows  # files are class-ordered; sample, don't slice


def timed(fn, cases):
    out = []
    for c in cases:
        t = time.perf_counter()
        r = fn(c)
        out.append((r, time.perf_counter() - t))
    return out  # [(result, latency_s)]


def lane(fn, cases, out, gate=None):
    if gate:
        gate.wait()
    t0 = time.perf_counter()
    res = timed(fn, cases)
    out.update(summarize(res, cases, time.perf_counter() - t0), raw=res)


def summarize(res, cases, total):
    lat = sorted(l for _, l in res)
    return {"n": len(res), "acc": sum(r["choice"] == c["label"] for (r, _), c in zip(res, cases)) / len(res),
            "p50_ms": 1000 * st.median(lat), "p95_ms": 1000 * lat[int(.95 * (len(lat) - 1))],
            "total_s": total, "throughput_rps": len(res) / total}


def shuffled(choices, rng):
    keys = list(choices)
    rng.shuffle(keys)
    return {k: choices[k] for k in keys}


def env_info(model, seed):
    def v(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True) if _has("nvidia-smi") else None
    return {"model": model, "seed": seed, "python": platform.python_version(), "gpu": gpu.stdout.strip() if gpu else None,
            "sglang": v("sglang"), "transformers": v("transformers"), "torch": v("torch"),
            "date": time.strftime("%Y-%m-%d %H:%M")}


def _has(cmd):
    from shutil import which
    return which(cmd) is not None


def stop_server(proc, timeout=30):
    """Reap the server even when graceful shutdown stalls."""
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def start_server(model, port, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as output:
        proc = subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(port)],
            stdout=output, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):  # first run downloads the model
            if proc.poll() is not None:
                tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                raise RuntimeError("server died, last log lines:\n" + "\n".join(tail))
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                    return proc
            except requests.RequestException:
                pass
            time.sleep(2)
        raise RuntimeError(f"server not healthy after 20 min, see {log}")
    except BaseException:
        stop_server(proc)
        raise


def sanity(jev, model, url, case):
    """One case through both endpoints. Tells us tonight, in seconds, whether the API assumptions hold."""
    out = {}
    for name in ("score", "generate"):
        j = Jev(model, url, tok=jev.tok)
        j.backend = name
        try:
            r = j.decide(case["task"], case["text"], case["choices"])
            out[name] = {"ok": j.backend == name, "probs": r["probabilities"], "sum": sum(r["probabilities"].values())}
        except Exception as e:  # report, don't die: a broken fallback must not block the main path
            out[name] = {"ok": False, "error": repr(e)[:300]}
    a, b = out["score"], out["generate"]
    if a["ok"] and b["ok"]:
        out["max_abs_diff"] = max(abs(a["probs"][k] - b["probs"][k]) for k in a["probs"])
    out["warn"] = (not a["ok"]) or abs(a.get("sum", 0) - 1) > 1e-3 or out.get("max_abs_diff", 0) > 0.05
    return out


def batched_run(jev, cases, batch, single):
    """Score lane with `batch` messages per request. Same task/choices for all (true for our data)."""
    c0, picks, per_batch = cases[0], [], []
    t0 = time.perf_counter()
    for i in range(0, len(cases), batch):
        t = time.perf_counter()
        picks += [r["choice"] for r in jev.decide_many(c0["task"], [c["text"] for c in cases[i:i + batch]], c0["choices"])]
        per_batch.append(time.perf_counter() - t)
    total = time.perf_counter() - t0
    return {"batch": batch, "n": len(cases), "acc": sum(p == c["label"] for p, c in zip(picks, cases)) / len(cases),
            "total_s": total, "throughput_rps": len(cases) / total, "batch_p50_ms": 1000 * st.median(per_batch),
            "agree_with_single": sum(p == s for p, s in zip(picks, single)) / len(cases)}


def run_model(model, url, a, domain="support"):
    rng = random.Random(a.seed)
    data_dir = Path(getattr(a, "data_dir", ROOT / "data"))
    dev, test = load(data_dir / f"{domain}_dev.jsonl", a.n), load(data_dir / f"{domain}_test.jsonl", a.n)
    jev = Jev(model, url)
    f_score = lambda c: (lambda r: {"choice": r["choice"], "probs": r["probabilities"]})(
        jev.decide(c["task"], c["text"], c["choices"]))
    f_gen = lambda c: generate(model, c["task"], c["text"], c["choices"], url)
    f_label = lambda c: generate_label(model, c["task"], c["text"], c["choices"], url)
    f_struct = lambda c: generate_structured(model, c["task"], c["text"], c["choices"], url)

    R_sanity = sanity(jev, model, url, test[0])
    print("sanity:", R_sanity, flush=True)
    print("warm-up", flush=True)
    for c in test[:10]:
        f_score(c); f_gen(c)
    try:
        f_struct(test[0])
        struct_ok = True
    except Exception as e:  # server without json_schema support: skip the lane, keep the rest
        print("structured lane skipped:", repr(e)[:200], flush=True)
        struct_ok = False

    single_ref = [{"probs": f_score(c)["probs"]} for c in test]
    R = {"env": {**env_info(model, a.seed), "domain": domain}, "sanity": R_sanity, "backend": jev.backend, "reps": a.reps, "solo": {"score": [], "generate": [], "structured": [], "label": []}, "batched": [],
         "concurrent": {"score": [], "generate": []}}
    R["lane_order"] = []
    R["cache_policy"] = {"solo": "randomized lanes; all cases primed before each lane (cache residency not guaranteed)",
                         "cold_per_request": getattr(a, "cold_cache", False)}
    for rep in range(a.reps):
        print(f"rep {rep + 1}/{a.reps}", flush=True)
        lanes = [("score", f_score), ("generate", f_gen), ("structured", f_struct), ("label", f_label)]
        rng.shuffle(lanes)
        R["lane_order"].append([name for name, _ in lanes])
        for name, fn in lanes:
            if name == "structured" and not struct_ok:
                continue
            for c in test:
                fn(c)  # explicitly prime identical cases before warm measurement
            o = {}; lane(fn, test, o); R["solo"][name].append(o)
        R["batched"].append(batched_run(jev, test, a.batch, single=[an.top(r["probs"])[0] for r in single_ref]))
        gate, o = threading.Barrier(2), {"score": {}, "generate": {}}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, fn, test, o[n], gate)
                       for n, fn in (("score", f_score), ("generate", f_gen))]
            for future in futures:
                future.result()  # propagate failures instead of writing partial benchmark results
        for n in o:
            R["concurrent"][n].append(o[n])
    R["backend"] = jev.backend  # may have flipped after the first request

    if getattr(a, "cold_cache", False):
        R["cold_cache"] = {}
        for name, fn in (("score", f_score), ("label", f_label), ("generate", f_gen)):
            measured = []
            for c in test:
                requests.post(f"{url}/flush_cache", timeout=120).raise_for_status()
                measured.extend(timed(fn, [c]))
            R["cold_cache"][name] = summarize(measured, test, sum(t for _, t in measured))

    # Probabilities for calibration: original option order, one pass each.
    rows = {}
    for name, cases in (("dev", dev), ("test", test)):
        rows[name] = [{"id": c["id"], "text": c["text"], "label": c["label"], "probs": f_score(c)["probs"]} for i, c in enumerate(cases)]
    # Preserve per-request evidence with stable IDs for later error and latency analysis.
    for lanes in (R["solo"], R["concurrent"]):
        for runs in lanes.values():
            for o in runs:
                raw = o.pop("raw", [])
                o["requests"] = [{"id": test[i]["id"], "label": test[i]["label"],
                                   "latency_s": latency, **result}
                                  for i, (result, latency) in enumerate(raw)]

    # Option-order bias: shuffle the option order (incl. `other`), compare with the original order.
    base = [an.top(r["probs"])[0] for r in rows["test"]]
    perms = []
    for _ in range(a.perms):
        pr = [f_score({**c, "choices": shuffled(c["choices"], rng)})["probs"] for c in test]
        perms.append(pr)
    picks = [[an.top(p)[0] for p in pr] for pr in perms]
    avg = [an.average_probs([rows["test"][i]["probs"]] + [pr[i] for pr in perms]) for i in range(len(test))]
    R["order_bias"] = {
        "perms": a.perms,
        "base_acc": sum(b == r["label"] for b, r in zip(base, rows["test"])) / len(base),
        "perm_acc": [sum(p == r["label"] for p, r in zip(pk, rows["test"])) / len(base) for pk in picks],
        "flip_rate": [an.flip_rate(base, pk) for pk in picks],
        "avg_acc": sum(an.top(p)[0] == r["label"] for p, r in zip(avg, rows["test"])) / len(base),
    }

    # Calibration: fit T and pick thresholds on dev only, report on test only.
    T = an.fit_temperature(rows["dev"])
    pick = an.pick_thresholds(rows["dev"], a.target_acc, T, getattr(a, "min_auto", 30), getattr(a, "policy_criterion", "observed"))
    raw_pick = an.pick_thresholds(rows["dev"], a.target_acc, 1.0, getattr(a, "min_auto", 30), getattr(a, "policy_criterion", "observed"))
    cal = {"T": T, "target_acc": a.target_acc,
           "reliability_raw": an.reliability(rows["test"]), "reliability_cal": an.reliability(rows["test"], T),
           "default_policy_test": an.evaluate_policy(rows["test"], 0.80, 0.20)}
    for k, pk, TT in (("tuned", pick, T), ("tuned_no_T", raw_pick, 1.0)):
        cal[k] = None if pk is None else {"min_p": pk[0], "min_margin": pk[1], "dev": pk[2],
                                          "test": an.evaluate_policy(rows["test"], pk[0], pk[1], TT)}
    R["calibration"] = cal
    R["test_rows"] = rows["test"]
    R["diagnostics"] = an.classification_report(rows["test"], T)
    R["risk_coverage"] = an.risk_coverage(rows["test"], T)
    R["policy_selection"] = {"min_auto": getattr(a, "min_auto", 30),
                              "criterion": getattr(a, "policy_criterion", "observed")}
    R["policy"] = None if pick is None else {
        "model": model, "temperature": T, "min_p": pick[0], "min_margin": pick[1],
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(dev[0]["task"], dev[0]["choices"]),
        "domain": domain, "target_acc": a.target_acc, "selection": R["policy_selection"]}
    challenge_path = data_dir / f"{domain}_challenge.jsonl"
    if challenge_path.exists():
        challenge = load(challenge_path)
        cr = [{**c, "id": c["id"], "probs": f_score(c)["probs"]} for c in challenge]
        R["challenge"] = {"source": "hand-authored synthetic; evaluation only",
                          "report": an.classification_report(cr, T),
                          "by_tag": {tag: an.classification_report([r for r in cr if tag in r["tags"]], T)
                                     for tag in sorted({tag for c in cr for tag in c["tags"]})}}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--url", help="use a running server instead of launching one (single model)")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--domains", nargs="+", default=["support", "news"], help="data/<domain>_{dev,test}.jsonl")
    ap.add_argument("--perms", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="messages per /v1/score call in the batched lane")
    ap.add_argument("--cold-cache", action="store_true", help="flush server KV cache before each solo request; dedicated server only")
    ap.add_argument("--min-auto", type=int, default=30)
    ap.add_argument("--policy-criterion", choices=["observed", "wilson"], default="observed",
                    help="require observed accuracy or its 95%% Wilson lower bound to reach target on dev")
    ap.add_argument("--target-acc", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, help="cap cases per split")
    ap.add_argument("--smoke", action="store_true", help="20 cases, 1 rep, 2 perms")
    ap.add_argument("--out", default=str(ROOT / "bench/results"))
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.reps, a.perms, a.batch = 20, 1, 2, 5
    if a.url and len(a.models) != 1:
        ap.error("--url requires exactly one model")
    if min(a.reps, a.batch, a.min_auto) < 1 or a.perms < 1 or (a.n is not None and a.n < 1):
        ap.error("reps, batch, min-auto, perms and n must be positive")
    datasets = [p for domain in a.domains for split in ("train", "dev", "test", "challenge")
                for p in [ROOT / f"data/{domain}_{split}.jsonl"] if p.exists()]
    for domain in a.domains:
        for split in ("dev", "test"):
            if not (ROOT / f"data/{domain}_{split}.jsonl").exists():
                ap.error(f"Missing data/{domain}_{split}.jsonl")
    with experiment(a.out, vars(a), datasets) as out:
        quality = audit(out.joinpath("data").glob("*.jsonl"))
        (out / "data-quality.json").write_text(json.dumps(quality, indent=2))
        if not quality["ok"]:
            raise ValueError("Dataset validation failed; see " + str(out / "data-quality.json"))
        print(f"data checks: {len(quality['warnings'])} warnings; see {out / 'data-quality.json'}", flush=True)
        run_experiment(a, out)
    print("experiment:", out, flush=True)


def run_experiment(a, out):
    a.data_dir = str(out / "data")
    for model in a.models:
        slug = model.replace("/", "__")
        proc = None if a.url else start_server(model, a.port, out / f"{slug}.server.log")
        try:
            for domain in a.domains:
                R = run_model(model, a.url or f"http://127.0.0.1:{a.port}", a, domain)
                f = out / f"{slug}__{domain}.json"
                f.write_text(json.dumps(R, indent=1))
                policy_file = out / f"{slug}__{domain}.policy.json"
                if R["policy"]:
                    policy_file.write_text(json.dumps(R["policy"], indent=2))
                else:
                    policy_file.unlink(missing_ok=True)
                from bench.report import write_report
                write_report(R, out / f"{slug}__{domain}.report.md", plot=not a.no_plot)
                print("wrote", f, flush=True)
                if a.smoke and R["sanity"]["warn"]:  # stop run.sh here instead of burning an hour on a broken API assumption
                    sys.exit(f"SANITY FAILED, see {f} -> sanity")
        finally:
            if proc:
                stop_server(proc)
    if not a.no_plot:
        from article import figures
        figures.HERE = out / "article"
        figures.IMG = figures.HERE / "img"
        figures.HERE.mkdir(parents=True, exist_ok=True)
        figures.main(out)


if __name__ == "__main__":
    main()

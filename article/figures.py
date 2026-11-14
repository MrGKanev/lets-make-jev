"""Every figure and table in the article. Static ones need no data; result ones read bench/results/*.json.

    python -m article.figures            # writes article/img/*.png and article/tables.md
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"
SCORE, GEN, GREY, STRUCT = "#0f766e", "#c2410c", "#64748b", "#6d28d9"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def save(fig, name):
    IMG.mkdir(exist_ok=True)
    fig.savefig(IMG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="white", ec=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=1.8))


def fig_flow():
    """Two lanes over one server: scoring reads logits once, generation decodes tokens then parses text."""
    fig, ax = plt.subplots(figsize=(12, 4.2)); ax.axis("off"); ax.set_xlim(-0.1, 11.4); ax.set_ylim(0, 4.2)
    ax.text(0.1, 3.85, "Score lane (Jev)", color=SCORE, weight="bold")
    ax.text(0.1, 1.75, "Generate lane (baseline)", color=GEN, weight="bold")
    s = [("question +\nlabels A/B/C/D", 0.1), ("prefill\n(one pass)", 2.5),
         ("logits of the\nfirst answer token", 4.9), ("softmax over\nA,B,C,D only", 7.3), ("probabilities\n+ route()", 9.4)]
    for t, x in s:
        box(ax, x, 2.45, 1.6, 1.0, t, SCORE)
    for (_, a), (_, b) in zip(s, s[1:]):
        arrow(ax, a + 1.65, 2.95, b - 0.05, 2.95, SCORE)
    g = [("question +\ncategory names", 0.1), ("prefill", 2.5), ("decode up to\n32 tokens", 4.9),
         ("free text", 7.3), ("parse the text,\nhope it matches", 9.4)]
    for t, x in g:
        box(ax, x, 0.35, 1.6, 1.0, t, GEN)
    for (_, a), (_, b) in zip(g, g[1:]):
        arrow(ax, a + 1.65, 0.85, b - 0.05, 0.85, GEN)
    save(fig, "fig_flow.png")


def fig_softmax():
    """Illustration only: made-up logits for the three labels, not a measurement."""
    import math
    names, logits = ["A  billing", "B  tech support", "C  account", "D  other"], [8.2, 5.5, 4.8, 2.0]
    e = [math.exp(x - max(logits)) for x in logits]; p = [v / sum(e) for v in e]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    a.barh(names[::-1], logits[::-1], color=GREY); a.set_title("Logits of the first answer token")
    b.barh(names[::-1], p[::-1], color=SCORE); b.set_title("Softmax over the four labels only"); b.set_xlim(0, 1)
    for i, v in enumerate(p[::-1]):
        b.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.suptitle("Illustrative numbers (not from our runs)", fontsize=9, color=GREY, y=-0.02)
    save(fig, "fig_softmax.png")


def load_results(d):
    results = []
    for path in sorted(Path(d).glob("*.json")):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "env" in value and "solo" in value:
            results.append(value)
    return results


def short(r):
    d = r["env"].get("domain", "support")
    return r["env"]["model"].split("/")[-1] + ("" if d == "support" else f"\n({d})")


def agg(runs, k):
    v = [x[k] for x in runs]
    return st.mean(v), min(v), max(v)


def grouped(ax, res, key, ylabel, title):
    """Bars: per model, four bars (score/generate x solo/concurrent). Whiskers = min..max over repetitions."""
    cols = [("solo", "score", SCORE, 1.0), ("concurrent", "score", SCORE, 0.5),
            ("solo", "generate", GEN, 1.0), ("concurrent", "generate", GEN, 0.5),
            ("solo", "structured", STRUCT, 1.0), ("solo", "label", "#2563eb", 1.0)]
    w = 0.16
    for j, (mode, lane, c, al) in enumerate(cols):
        xs, ys, lo, hi = [], [], [], []
        for i, r in enumerate(res):
            if not r[mode].get(lane):
                continue
            m, a, b = agg(r[mode][lane], key)
            xs.append(i + (j - 2) * w); ys.append(m); lo.append(m - a); hi.append(b - m)
        ax.bar(xs, ys, w, color=c, alpha=al, yerr=[lo, hi], capsize=2, label=f"{lane} ({mode})")
    ax.set_xticks(range(len(res))); ax.set_xticklabels([short(r) for r in res]); ax.set_ylabel(ylabel); ax.set_title(title)


def fig_latency(res):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    grouped(axs[0], res, "p50_ms", "ms", "Median latency per request")
    grouped(axs[1], res, "p95_ms", "ms", "p95 latency per request")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig_latency.png")


def fig_accuracy(res):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, (name, c, get) in enumerate([
        ("generate", GEN, lambda r: agg(r["solo"]["generate"], "acc")[0]),
        ("structured JSON", STRUCT, lambda r: agg(r["solo"]["structured"], "acc")[0] if r["solo"].get("structured") else float("nan")),
        ("one-token label", "#2563eb", lambda r: agg(r["solo"]["label"], "acc")[0] if r["solo"].get("label") else float("nan")),
        ("score", SCORE, lambda r: agg(r["solo"]["score"], "acc")[0]),
        ("score, order-averaged", "#14b8a6", lambda r: r["order_bias"]["avg_acc"])]):
        ax.bar([i + (j - 2) * 0.17 for i in range(len(res))], [get(r) for r in res], 0.17, color=c, label=name)
    ax.set_xticks(range(len(res))); ax.set_xticklabels([short(r) for r in res])
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy on test"); ax.legend(frameon=False, fontsize=8); ax.set_title("Accuracy on test cases")
    save(fig, "fig_accuracy.png")


def fig_reliability(res):
    fig, axs = plt.subplots(1, len(res), figsize=(4.6 * len(res), 4.2), squeeze=False)
    for ax, r in zip(axs[0], res):
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
        for key, c, lab in (("reliability_raw", GEN, "raw"), ("reliability_cal", SCORE, f"T={r['calibration']['T']:.2f}")):
            table, ece = r["calibration"][key]
            ax.plot([t[3] for t in table], [t[4] for t in table], "o-", color=c, label=f"{lab} (ECE {ece:.3f})")
        ax.set_xlabel("stated confidence (top probability)"); ax.set_ylabel("actual accuracy")
        ax.set_title(short(r)); ax.legend(frameon=False, fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig_reliability.png")


def fig_order(res):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, r in enumerate(res):
        o = r["order_bias"]
        ax.scatter([i] * len(o["flip_rate"]), o["flip_rate"], color=SCORE, zorder=3)
        ax.plot([i - .2, i + .2], [st.mean(o["flip_rate"])] * 2, color="black")
    ax.set_xticks(range(len(res))); ax.set_xticklabels([short(r) for r in res]); ax.set_ylim(0, None)
    ax.set_ylabel("decisions that change"); ax.set_title("Shuffling the option order (dot = one shuffle, bar = mean)")
    save(fig, "fig_order.png")


def tables(res):
    L = ["## Latency and accuracy (test, mean over repetitions)\n",
         "| model | lane | mode | acc | p50 ms | p95 ms | req/s |", "|---|---|---|---|---|---|---|"]
    for r in res:
        for mode in ("solo", "concurrent"):
            for lane in ("score", "generate", "structured", "label"):
                if not r[mode].get(lane):
                    continue
                g = lambda k: agg(r[mode][lane], k)[0]
                L.append(f"| {short(r)} | {lane} | {mode} | {g('acc'):.3f} | {g('p50_ms'):.0f} | {g('p95_ms'):.0f} | {g('throughput_rps'):.1f} |")
    L += ["\n## Batched scoring (test; one /v1/score call carries several messages)\n",
          "| model | batch | acc | req/s | ms per batch (p50) | same decision as one-by-one |", "|---|---|---|---|---|---|"]
    for r in res:
        for b in r.get("batched", [])[:1] and [r["batched"][0]]:
            m = lambda k: st.mean(x[k] for x in r["batched"])
            L.append(f"| {short(r)} | {b['batch']} | {m('acc'):.3f} | {m('throughput_rps'):.1f} | {m('batch_p50_ms'):.0f} | {m('agree_with_single'):.1%} |")
    L += ["\n## Option-order bias (test)\n", "| model | acc (original order) | acc (shuffled, mean) | decisions that change | acc (order-averaged) |", "|---|---|---|---|---|"]
    for r in res:
        o = r["order_bias"]
        L.append(f"| {short(r)} | {o['base_acc']:.3f} | {st.mean(o['perm_acc']):.3f} | {st.mean(o['flip_rate']):.3f} | {o['avg_acc']:.3f} |")
    L += ["\n## Automation policy (thresholds chosen on dev, reported on test)\n",
          "| model | policy | min_p | min_margin | automated | accuracy when automated |", "|---|---|---|---|---|---|"]
    for r in res:
        c = r["calibration"]
        d = c["default_policy_test"]
        L.append(f"| {short(r)} | untuned 0.80/0.20 | 0.80 | 0.20 | {d['coverage']:.1%} | {_pct(d['auto_acc'])} |")
        for k, lab in (("tuned_no_T", "tuned on dev"), ("tuned", f"tuned on dev + temperature T={c['T']:.2f}")):
            t = c[k]
            L.append(f"| {short(r)} | {lab} | " + (f"{t['min_p']:.2f} | {t['min_margin']:.1f} | {t['test']['coverage']:.1%} | {_pct(t['test']['auto_acc'])} |"
                                                    if t else f"- | - | unreachable at {c['target_acc']:.0%} | - |"))
    L += ["\n## Environment\n"] + [f"- {short(r)}: {json.dumps(r['env'])}" for r in res]
    (HERE / "tables.md").write_text("\n".join(L) + "\n")


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def main(results_dir=HERE.parent / "bench/results"):
    global HERE, IMG
    results_dir = Path(results_dir)
    if not (results_dir / "manifest.json").exists():
        completed = [p.parent for p in results_dir.glob("*/manifest.json")
                     if json.loads(p.read_text()).get("status") == "complete"]
        if completed:
            results_dir = max(completed, key=lambda p: p.name)
    if (results_dir / "manifest.json").exists():
        HERE = results_dir / "article"
        IMG = HERE / "img"
        HERE.mkdir(parents=True, exist_ok=True)
    fig_flow(); fig_softmax()
    res = load_results(results_dir)
    if res:
        fig_latency(res); fig_accuracy(res); fig_reliability(res); fig_order(res); tables(res)
    print(f"figures: static + {'result' if res else 'no result'} figures in {IMG}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "bench/results")

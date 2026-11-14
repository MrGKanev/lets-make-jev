"""Pure functions over scored rows: {"label": key, "probs": {key: p}}. No server, no GPU."""
import math

from jev.decide import route


def temper(probs, T):
    """Temperature scaling on probabilities == scaling the logits (log p = logit - const)."""
    w = {k: max(p, 1e-12) ** (1 / T) for k, p in probs.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def top(probs):
    k = max(probs, key=probs.get)
    return k, probs[k]


def fit_temperature(rows):
    """Grid search T in [0.25, 5] minimising NLL of the true label."""
    def nll(T):
        return -sum(math.log(max(temper(r["probs"], T)[r["label"]], 1e-12)) for r in rows)
    return min((t / 20 for t in range(5, 101)), key=nll)


def reliability(rows, T=1.0, bins=10):
    """Returns (table, ece). table rows: (lo, hi, n, mean_conf, acc)."""
    got = [(top(temper(r["probs"], T)), r["label"]) for r in rows]
    table, ece = [], 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, k == y) for (k, p), y in got if lo <= p < hi or (b == bins - 1 and p == 1)]
        if sel:
            conf, acc = sum(p for p, _ in sel) / len(sel), sum(ok for _, ok in sel) / len(sel)
            ece += len(sel) / len(rows) * abs(conf - acc)
            table.append((lo, hi, len(sel), conf, acc))
    return table, ece


def evaluate_policy(rows, min_p, min_margin, T=1.0):
    """Coverage = share routed to 'auto'; auto_acc = accuracy on that share."""
    auto = ok = 0
    for r in rows:
        res = {"choice": top(r["probs"])[0], "probabilities": temper(r["probs"], T)}
        if route(res, min_p, min_margin) == "auto":
            auto += 1
            ok += res["choice"] == r["label"]
    return {"coverage": auto / len(rows), "auto_acc": ok / auto if auto else None, "n_auto": auto, "n_correct": ok, "accuracy_ci95": wilson(ok, auto)}


def pick_thresholds(rows, target_acc=0.95, T=1.0, min_auto=30, criterion="observed"):
    """Max coverage subject to auto accuracy >= target on these (dev) rows. None if unreachable."""
    if criterion not in ("observed", "wilson") or min_auto < 1:
        raise ValueError("Invalid threshold selection criterion or sample minimum")
    best = None
    for p in (x / 100 for x in range(50, 100)):
        for m in (0, .1, .2, .3, .4, .5):
            e = evaluate_policy(rows, p, m, T)
            if e["auto_acc"] is not None and e["auto_acc"] >= target_acc and e["n_auto"] >= min_auto and (criterion == "observed" or e["accuracy_ci95"][0] >= target_acc):
                if best is None or e["coverage"] > best[2]["coverage"]:
                    best = (p, m, e)
    return best


def flip_rate(base, other):
    return sum(a != b for a, b in zip(base, other)) / len(base)


def average_probs(prob_lists):
    keys = prob_lists[0]
    return {k: sum(p[k] for p in prob_lists) / len(prob_lists) for k in keys}


def wilson(ok, n, z=1.96):
    """Two-sided 95% Wilson interval; descriptive, not a deployment guarantee."""
    if not n:
        return [0.0, 1.0]
    p = ok / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - radius), min(1, center + radius)]


def classification_report(rows, T=1.0):
    keys = list(rows[0]["probs"])
    matrix = {y: {p: 0 for p in keys} for y in keys}
    errors = []
    for r in rows:
        pred, conf = top(temper(r["probs"], T))
        matrix[r["label"]][pred] += 1
        if pred != r["label"]:
            errors.append({**r, "prediction": pred, "confidence": conf})
    metrics = {}
    for k in keys:
        tp = matrix[k][k]
        support = sum(matrix[k].values())
        predicted = sum(matrix[y][k] for y in keys)
        precision, recall = tp / predicted if predicted else 0, tp / support if support else 0
        metrics[k] = {"precision": precision, "recall": recall,
                      "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                      "support": support}
    return {"confusion_matrix": matrix, "per_class": metrics,
            "macro_f1": sum(m["f1"] for m in metrics.values()) / len(keys),
            "errors": sorted(errors, key=lambda r: r["confidence"], reverse=True)}


def risk_coverage(rows, T=1.0):
    return [{"min_p": p / 100, **evaluate_policy(rows, p / 100, 0, T)} for p in range(0, 101, 2)]

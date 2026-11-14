"""Portable diagnostics and automation tradeoff figure for each benchmark."""
import json
from pathlib import Path


def write_report(result, path, plot=True):
    path = Path(path)
    d = result["diagnostics"]
    lines = [f'# Decision diagnostics: {result["env"]["model"]} / {result["env"]["domain"]}',
             '', 'Test set only; thresholds and temperature selected on dev.',
             '95% Wilson intervals describe binomial sampling uncertainty; they do not guarantee future accuracy.',
             f'Policy selection: `{json.dumps(result["policy_selection"])}`.', '',
             '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    for k, m in d['per_class'].items():
        lines.append(f'| {k} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["f1"]:.3f} | {m["support"]} |')
    lines += ['', '## Latency comparison', '',
              'Warm runs prime all cases before measurement. Lane order is shuffled per repetition. Cache residency is not guaranteed.',
              '| Lane | Cache | Repetition | N | Accuracy | p50 ms | p95 ms | req/s |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, runs in result['solo'].items():
        for i, r in enumerate(runs):
            lines.append(f'| {name} | primed | {i+1} | {r["n"]} | {r["acc"]:.3f} | {r["p50_ms"]:.1f} | {r["p95_ms"]:.1f} | {r["throughput_rps"]:.2f} |')
    for name, r in result.get('cold_cache', {}).items():
        lines.append(f'| {name} | flushed before each request | 1 | {r["n"]} | {r["acc"]:.3f} | {r["p50_ms"]:.1f} | {r["p95_ms"]:.1f} | {r["throughput_rps"]:.2f} |')
    lines += ['', 'Cold-cache request time excludes cache-flush time. Without `--cold-cache`, cold rows are not measured.', '']
    keys = list(d['confusion_matrix'])
    lines += ['', 'Confusion matrix: rows = true class, columns = predicted class.', '',
              '| True / predicted | ' + ' | '.join(keys) + ' |', '|---|' + '---:|' * len(keys)]
    for k, counts in d['confusion_matrix'].items():
        lines.append('| ' + k + ' | ' + ' | '.join(str(counts[p]) for p in keys) + ' |')
    lines += ['', '## Automation', '', '| Policy | Automated / all | Accuracy | 95% CI |', '|---|---:|---:|---|']
    for key in ('tuned', 'tuned_no_T'):
        policy = result['calibration'][key]
        if policy:
            t = policy['test']; lo, hi = t['accuracy_ci95']
            acc = 'n/a' if t['auto_acc'] is None else f'{t["auto_acc"]:.1%}'
            lines.append(f'| {key} | {t["n_auto"]} / {len(result["test_rows"])} | {acc} | {lo:.1%}–{hi:.1%} |')
        else:
            lines.append(f'| {key} | No eligible dev threshold | n/a | n/a |')
    lines += ['', '## Highest-confidence errors', '']
    for r in d['errors']:
        lines.append(f'- `{r["id"]}`: expected **{r["label"]}**, predicted **{r["prediction"]}** ({r["confidence"]:.1%}). ' + json.dumps(r['text'], ensure_ascii=False))
    if 'challenge' in result:
        lines += ['', '## Synthetic challenge set', '',
                  'Hand-authored diagnostic examples, excluded from training, calibration and headline test metrics.',
                  'Multi-intent cases have an explicit primary intent; their labels are editorial choices.', '',
                  '| Tag | N | Accuracy |', '|---|---:|---:|']
        for tag, rep in result['challenge']['by_tag'].items():
            n = sum(m['support'] for m in rep['per_class'].values())
            lines.append(f'| {tag} | {n} | {1-len(rep["errors"])/n:.1%} |')
        lines += ['', 'Challenge errors:', '']
        for r in result['challenge']['report']['errors']:
            lines.append(f'- `{r["id"]}`: {r["label"]} → {r["prediction"]} ({r["confidence"]:.1%}): ' + json.dumps(r['text'], ensure_ascii=False))
    image_path = path.with_suffix('.risk.png')
    if plot:
        plot_risk(result['risk_coverage'], image_path)
    lines += ['', '## Coverage and error tradeoff', '',
              'Exploratory sweep on test, with margin = 0 and `other` always sent to review. Do not select deployment thresholds from this curve.',
              f'![Coverage versus error rate]({image_path.name})' if plot else 'Plot omitted with --no-plot; numeric sweep is in the JSON.']
    path.write_text('\n'.join(lines) + '\n')


def plot_risk(curve, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points = sorted((r for r in curve if r['n_auto']), key=lambda r: r['coverage'])
    fig, ax = plt.subplots(figsize=(7, 4))
    if points:
        x = [r['coverage'] for r in points]
        ax.plot(x, [1-r['auto_acc'] for r in points], color='#0f766e')
        ax.fill_between(x, [1-r['accuracy_ci95'][1] for r in points],
                        [1-r['accuracy_ci95'][0] for r in points], alpha=.2, color='#0f766e', label='95% Wilson interval')
        ax.legend()
    else:
        ax.text(.5, .5, 'No automated decisions', ha='center')
    ax.set(xlabel='Share automated (all test cases)', ylabel='Error rate among automated cases',
           xlim=(0, 1), ylim=(0, 1), title='Coverage / error tradeoff (test; exploratory)')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

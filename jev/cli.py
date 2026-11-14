"""JSONL prediction and human review workflows: python -m jev --help."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

from data.validate import validate_row
from jev.decide import Jev


def read_jsonl(path):
    source = sys.stdin if str(path) == '-' else open(path, encoding='utf-8')
    try:
        rows = []
        for line, text in enumerate(source, 1):
            if text.strip():
                try:
                    row = json.loads(text)
                    if not isinstance(row, dict):
                        raise ValueError('row must be an object')
                except ValueError as exc:
                    raise ValueError(f'{path}:{line}: {exc}') from exc
                rows.append(row)
        return rows
    finally:
        if source is not sys.stdin:
            source.close()


def write_jsonl(path, rows):
    output = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
    if str(path) == '-':
        sys.stdout.write(output)
    else:
        with open(path, 'x', encoding='utf-8') as target:
            target.write(output)


def indexed(rows):
    result = {}
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('Every record needs a nonempty string id')
        if row['id'] in result:
            raise ValueError(f'Duplicate id: {row["id"]}')
        result[row['id']] = row
    return result


def fingerprint(row):
    payload = [row[k] for k in ('id', 'task', 'text')] + [list(row['choices'].items()), row['choice'], row['probabilities']]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def review_queue(predictions):
    indexed(predictions)
    return [{**r, 'source_fingerprint': fingerprint(r), 'review_label': None, 'review_notes': ''}
            for r in predictions if r['route'] == 'review']


def apply_reviews(predictions, reviews):
    originals = indexed(predictions)
    indexed(reviews)
    labeled = []
    for review in reviews:
        if review['id'] not in originals:
            raise ValueError(f'Unknown review id: {review["id"]}')
        original = originals[review['id']]
        if review.get('source_fingerprint') != fingerprint(original) or fingerprint(review) != fingerprint(original):
            raise ValueError(f'Review source changed: {review["id"]}')
        label = review.get('review_label')
        if label is None:
            continue
        if not isinstance(label, str) or label not in original['choices']:
            raise ValueError(f'Invalid review_label: {review["id"]}')
        if not isinstance(review.get('review_notes', ''), str):
            raise ValueError('review_notes must be text')
        labeled.append({k: original[k] for k in ('id', 'task', 'text', 'choices')})
        labeled[-1].update(label=label, review={'original_choice': original['choice'],
                         'notes': review.get('review_notes', ''), 'source_fingerprint': fingerprint(original)})
    if not labeled:
        raise ValueError('No completed reviews; fill review_label with an allowed category')
    agreed = sum(originals[r['id']]['choice'] == r['label'] for r in labeled)
    return labeled, {'n_reviewed': len(labeled), 'n_correct_original': agreed,
                     'original_accuracy_on_reviewed': agreed / len(labeled),
                     'scope': 'Human-reviewed subset only; not representative of all traffic'}


def check_outputs(paths):
    paths = [str(p) for p in paths if p is not None]
    names = ['-' if p == '-' else str(Path(p).resolve()) for p in paths]
    if len(set(names)) != len(names):
        raise ValueError('Output paths must be distinct')
    for path in paths:
        if path != '-' and Path(path).exists():
            raise ValueError(f'Output already exists: {path}; choose a new file')


def predict(args):
    check_outputs([args.output, args.review])
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError('Input is empty')
    for i, row in enumerate(rows):
        validate_row(row, require_label=False)
        if 'label' in row and row['label'] not in row['choices']:
            raise ValueError(f'Invalid reference label at row {i+1}')
        row.setdefault('id', f'input:{i+1}')
    indexed(rows)
    engine = Jev(args.model, args.url, policy=args.policy)
    predictions, i = [], 0
    while i < len(rows):
        first = rows[i]
        group = [first]
        for row in rows[i+1:i+args.batch]:
            if row['task'] != first['task'] or list(row['choices'].items()) != list(first['choices'].items()):
                break
            group.append(row)
        results = engine.decide_many(first['task'], [r['text'] for r in group], first['choices']) if args.batch > 1 else [engine.decide(first['task'], first['text'], first['choices'])]
        if len(results) != len(group):
            raise ValueError(f"Expected {len(group)} decisions, received {len(results)}")
        for row, result in zip(group, results):
            predictions.append({**row, **result, 'model': args.model})
        i += len(group)
    write_jsonl(args.output, predictions)
    if args.review:
        write_jsonl(args.review, review_queue(predictions))
    labeled = [r for r in predictions if 'label' in r]
    summary = {'n': len(predictions), 'n_review': sum(r['route'] == 'review' for r in predictions),
               'n_labeled': len(labeled),
               'accuracy': sum(r['choice'] == r['label'] for r in labeled) / len(labeled) if labeled else None}
    print(json.dumps(summary), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pred = commands.add_parser('predict', help='score JSONL; optional labels enable re-evaluation')
    pred.add_argument('--input', required=True)
    pred.add_argument('--output', required=True)
    pred.add_argument('--model', required=True)
    pred.add_argument('--url', default='http://127.0.0.1:30000')
    pred.add_argument('--policy')
    pred.add_argument('--batch', type=int, default=1)
    pred.add_argument('--review', help='also export review queue')
    review = commands.add_parser('review', help='export or apply human corrections')
    sub = review.add_subparsers(dest='action', required=True)
    export = sub.add_parser('export')
    export.add_argument('--predictions', required=True)
    export.add_argument('--output', required=True)
    apply = sub.add_parser('apply')
    apply.add_argument('--predictions', required=True)
    apply.add_argument('--reviews', required=True)
    apply.add_argument('--output', required=True, help='new labeled JSONL for re-evaluation')
    args = parser.parse_args(argv)
    try:
        if args.command == 'predict':
            if args.batch < 1:
                raise ValueError('--batch must be positive')
            predict(args)
        elif args.action == 'export':
            check_outputs([args.output])
            write_jsonl(args.output, review_queue(read_jsonl(args.predictions)))
        else:
            check_outputs([args.output])
            if args.predictions == args.reviews == '-':
                raise ValueError('Predictions and reviews cannot both read stdin')
            labeled, summary = apply_reviews(read_jsonl(args.predictions), read_jsonl(args.reviews))
            write_jsonl(args.output, labeled)
            print(json.dumps(summary), file=sys.stderr)
    except (ValueError, OSError, KeyError, TypeError, requests.RequestException) as exc:
        parser.exit(2, f'error: {exc}\n')
    return 0

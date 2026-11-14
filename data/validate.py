"""CPU-only schema, duplicate and split leakage checks for JSONL datasets."""
import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def read_rows(path):
    rows = []
    with open(path, encoding='utf-8') as source:
        for line, text in enumerate(source, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
                validate_row(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{line}: {exc}') from exc
            rows.append(row)
    return rows


def validate_row(row, require_label=True):
    if not isinstance(row, dict):
        raise ValueError('row must be an object')
    for key in ('task', 'text'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'{key} must be nonempty text')
    choices = row.get('choices')
    if not isinstance(choices, dict) or not 2 <= len(choices) <= 10:
        raise ValueError('choices must contain 2 to 10 categories')
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip() for k, v in choices.items()):
        raise ValueError('choice keys and descriptions must be nonempty strings')
    if require_label and row.get('label') not in choices:
        raise ValueError('label must be an allowed category')
    if 'id' in row and (not isinstance(row['id'], str) or not row['id'].strip()):
        raise ValueError('id must be nonempty text')


def audit(paths):
    errors, warnings, files = [], [], {}
    occurrences, definitions, global_ids = defaultdict(list), {}, {}
    for path in sorted(map(Path, paths)):
        try:
            domain, split = path.stem.rsplit('_', 1)
            rows = read_rows(path)
        except ValueError as exc:
            errors.append(str(exc)); continue
        files[str(path)] = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if not rows:
            errors.append(f'{path}: empty dataset')
        seen, ids = {}, set()
        for i, row in enumerate(rows, 1):
            location = f'{path}:{i}'
            definition = (row['task'], list(row['choices'].items()))
            if domain in definitions and definitions[domain] != definition:
                errors.append(f'{location}: task or category definitions/order differ within domain {domain}')
            definitions.setdefault(domain, definition)
            text = normalized(row['text'])
            if text in seen:
                errors.append(f'{location}: duplicate normalized text (first at row {seen[text]})')
            seen[text] = i
            if 'id' in row:
                if row['id'] in ids:
                    errors.append(f'{location}: duplicate id {row["id"]}')
                ids.add(row['id'])
                identity = (domain, row['id'])
                if identity in global_ids and global_ids[identity] != str(path):
                    errors.append(f'{location}: duplicate id across domain files: {row["id"]}')
                global_ids[identity] = str(path)
            occurrences[text].append((domain, split, location))
    for entries in occurrences.values():
        splits = {split for _, split, _ in entries}
        if len(splits) > 1:
            errors.append('Cross-split text leakage: ' + ', '.join(loc for _, _, loc in entries))
        elif len({domain for domain, _, _ in entries}) > 1:
            warnings.append('Shared text across domains in same split: ' + ', '.join(loc for _, _, loc in entries))
    return {'ok': not errors, 'errors': errors, 'warnings': warnings, 'files': files,
            'normalization': 'Unicode NFKC, casefold, collapsed whitespace; exact normalized matches only'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    paths = [p for split in ('train', 'dev', 'test', 'challenge') for p in args.data_dir.glob(f'*_{split}.jsonl')]
    result = audit(paths)
    if not paths:
        result['ok'] = False; result['errors'].append('No dataset files found')
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(output, encoding='utf-8')
    else:
        print(output, end='')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Isolated experiment directories with configuration and dataset snapshots. No Git access."""
import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def experiment(root, config, datasets):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8]
    out = Path(root).resolve() / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / 'data').mkdir()
    manifest = {'run_id': run_id, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'config': dict(config), 'datasets': {}}
    try:
        for source in datasets:
            source = Path(source)
            target = out / 'data' / source.name
            shutil.copyfile(source, target)
            manifest['datasets'][source.name] = {'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                                                'snapshot': str(target.relative_to(out))}
        write_json(out / 'manifest.json', manifest)
        yield out
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        manifest['status'] = 'complete'
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'manifest.json', manifest)

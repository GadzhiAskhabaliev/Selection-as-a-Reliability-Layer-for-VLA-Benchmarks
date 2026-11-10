"""One-shot read-only Kaggle status/output check; does not schedule or launch jobs."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ReadAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        kwargs['timeout'] = (15, 40)
        return super().send(request, **kwargs)


original_init = requests.Session.__init__


def session_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.mount('https://', ReadAdapter(max_retries=Retry(total=2, backoff_factor=1, allowed_methods=None)))


requests.Session.__init__ = session_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    from kaggle import api
    status = json.loads(str(api.kernels_status(args.kernel)))
    record = {'checked_at': datetime.now().astimezone().isoformat(), 'kernel': args.kernel, **status}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'status.json').write_text(json.dumps(record, indent=2))
    with (args.output / 'status_history.jsonl').open('a') as history:
        history.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
    if status['status'] in ('RUNNING', 'QUEUED', 'CANCEL_REQUESTED'):
        return
    dest = args.output / 'output'
    print(api.kernels_output(args.kernel, path=str(dest), force=True), flush=True)
    for name in ('failure.json', 'smoke_checks.json', 'result.json'):
        file = dest / name
        if file.exists():
            print(name, file.read_text()[:12000], flush=True)
    for file in dest.glob('*.log'):
        entries = json.loads(file.read_text())
        print('LOG TAIL', flush=True)
        for entry in entries[-16:]:
            print(entry.get('time'), entry.get('data', '').rstrip(), flush=True)


if __name__ == '__main__':
    main()

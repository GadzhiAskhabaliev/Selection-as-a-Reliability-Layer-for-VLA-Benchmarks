"""Read-only SHA-256 check of the published experiment archive."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / 'artifacts/checksums.json').read_text())
    for relative, expected in index['sha256'].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Archive checksum mismatch: {relative}')
    print(f'PASS: {len(index["sha256"])} archived files match SHA-256 checksums')


if __name__ == '__main__':
    main()

"""Upload the release directory built by `build_hf_release.py` to the Hugging Face Hub.

Authentication uses the ``HF_TOKEN`` environment variable if set, otherwise the token stored by
``huggingface-cli login``. The release manifest is checked against the files before uploading.

Usage:
    python scripts/upload_to_hf.py [--repo-id USER/NAME] [--repo-type model|dataset] [--folder hf_release]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def check_manifest(folder: Path) -> int:
    """Verify that every file listed in MANIFEST.json exists with the recorded SHA-256.

    Returns:
        Number of verified files.

    Raises:
        ValueError: If a file is missing or its hash differs.
    """
    entries = json.loads((folder / 'MANIFEST.json').read_text())
    for entry in entries:
        path = folder / entry['path']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'{entry["path"]} does not match MANIFEST.json; rebuild the release')
    return len(entries)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--repo-id', default='hackhackhack66666/vla-test-time-and-multinet-audit')
    p.add_argument('--repo-type', default='model', choices=('model', 'dataset'))
    p.add_argument('--folder', type=Path, default=ROOT / 'hf_release')
    p.add_argument('--message', default='Upload VLA test-time selection and MultiNet audit artifacts')
    args = p.parse_args()

    files = check_manifest(args.folder)
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    api.create_repo(args.repo_id, repo_type=args.repo_type, exist_ok=True)
    commit = api.upload_folder(folder_path=str(args.folder), repo_id=args.repo_id, repo_type=args.repo_type,
                               commit_message=args.message)
    print(f'Uploaded {files} files to {args.repo_type} {args.repo_id}: {commit.commit_url}')


if __name__ == '__main__':
    main()

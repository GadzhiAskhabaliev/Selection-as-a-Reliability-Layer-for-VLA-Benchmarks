"""Read-only runtime and public data audit; no inference or baseline scores."""
import concurrent.futures
import importlib.metadata
import json
import platform
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

COMMIT = "546a1c77f9a4cc0544c5e4b41565949778d32441"
OUT = Path('/kaggle/working')


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def inspect_dataset(name):
    prefix = 'robotics/' + name + '/'
    query = urllib.parse.urlencode({'prefix': prefix, 'maxResults': 1000,
                                   'fields': 'items(name,size),nextPageToken'})
    try:
        listing = fetch_json('https://storage.googleapis.com/storage/v1/b/gresearch/o?' + query)
        infos = [item for item in listing.get('items', [])
                 if item['name'].endswith('/dataset_info.json')]
        result = {'dataset': name, 'listing_truncated': bool(listing.get('nextPageToken')),
                  'metadata': []}
        for item in infos:
            url = 'https://storage.googleapis.com/gresearch/' + item['name']
            info = fetch_json(url)
            result['metadata'].append({'url': url, 'info': info})
        return result
    except Exception as exc:
        return {'dataset': name, 'error': str(exc)}


def main():
    report = {'kind': 'preflight_only', 'multinet_commit': COMMIT,
              'python': platform.python_version(), 'packages': {}}
    for package in ('torch', 'torchvision', 'tensorflow', 'tensorflow-datasets',
                    'transformers', 'timm', 'accelerate', 'numpy'):
        try:
            report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True)
    report['gpu'] = gpu.stdout.strip()
    import torch
    report['cuda'] = torch.version.cuda
    report['bf16_supported'] = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    root = f'https://raw.githubusercontent.com/ManifoldRG/MultiNet/{COMMIT}/'
    reference = fetch_json(root + 'src/eval/profiling/openvla/results/openvla_openx_eval_results.json')
    report['published_dataset_count'] = len(reference)
    report['published_steps_total'] = sum(r['num_timesteps'] for r in reference.values())
    model = fetch_json('https://huggingface.co/api/models/openvla/openvla-7b')
    report['model'] = {'id': model['id'], 'sha': model['sha'], 'gated': model.get('gated')}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        report['datasets'] = list(pool.map(inspect_dataset, reference))
    (OUT / 'preflight.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'datasets'}, indent=2))
    for item in report['datasets']:
        print(item['dataset'], 'metadata_versions=', len(item.get('metadata', [])),
              'error=', item.get('error'))


if __name__ == '__main__':
    main()

"""Full Jaco Play test baseline, with MultiNet v0.1 adapters; no CS.

The full 20-dataset reproduction is NOT complete when this job completes.
Kaggle runtime adaptations (dtype, attention, placement) are logged explicitly.
Weights and downloaded data live in /kaggle/temp, not notebook outputs.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import types
import urllib.request

COMMIT = "546a1c77f9a4cc0544c5e4b41565949778d32441"
MODEL = "openvla/openvla-7b"
REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
DATASET = "jaco_play"
DATA_VERSION = "0.1.0"
SPLIT = "test"
EXPECTED_EPISODES = 109
EXPECTED_STEPS = 7838
SEED = 7
OUT = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/multinet_baseline')


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as dest:
                while chunk := response.read(1024 * 1024):
                    dest.write(chunk)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def mse(prediction, target):
    if not prediction or len(prediction) != len(target):
        raise ValueError('Action dimensions differ or are empty')
    if not all(math.isfinite(float(v)) for v in [*prediction, *target]):
        raise ValueError('Non-finite action')
    return sum((float(p) - float(t)) ** 2 for p, t in zip(prediction, target)) / len(target)


def aggregate(errors):
    if not errors or not all(math.isfinite(v) and v >= 0 for v in errors):
        raise ValueError('Expected finite, non-negative errors from a nonempty run')
    lo, hi = min(errors), max(errors)
    # MultiNet's min-max NAMSE is undefined for a zero range. Report this
    # explicitly rather than silently substituting a score or dividing by zero.
    return {'num_timesteps': len(errors), 'amse': sum(errors) / len(errors),
            'namse': sum((v - lo) / (hi - lo) for v in errors) / len(errors) if hi > lo else None,
            'namse_defined': hi > lo, 'min_mse': lo, 'max_mse': hi}


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def bootstrap():
    # Kaggle's current Python 3.13 cannot use the original tokenizers wheels.
    # Do not mix its Torch 2.11 / Transformers 5 stack with the 2024 model.
    if os.environ.get('GDZ_BASELINE_ENV_READY') == '1':
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError('Baseline environment must use Python 3.11')
        return
    uv = shutil.which('uv')
    if uv is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
        uv = shutil.which('uv')
    if uv is None:
        raise RuntimeError('uv installation did not provide an executable')
    os.environ['UV_CACHE_DIR'] = str(SCRATCH / 'uv_cache')
    os.environ['UV_PYTHON_INSTALL_DIR'] = str(SCRATCH / 'python')
    env_dir = SCRATCH / 'env'
    subprocess.run([uv, 'venv', '--python', '3.11', str(env_dir)], check=True)
    python = str(env_dir / 'bin/python')
    subprocess.run([uv, 'pip', 'install', '--python', python,
                    '--index-url', 'https://download.pytorch.org/whl/cu121',
                    'torch==2.2.0', 'torchvision==0.17.0'], check=True)
    subprocess.run([uv, 'pip', 'install', '--python', python,
                    '--only-binary', 'tokenizers,sentencepiece',
                    'numpy==1.26.4', 'transformers==4.40.1', 'tokenizers==0.19.1',
                    'timm==0.9.10', 'huggingface-hub==0.23.5', 'accelerate==0.30.1',
                    'tensorflow-cpu==2.15.1', 'tensorflow-datasets==4.9.3',
                    'tensorflow-metadata==1.15.0', 'protobuf==4.25.8',
                    'sentencepiece==0.2.0', 'safetensors==0.4.5'], check=True)
    frozen = subprocess.check_output([uv, 'pip', 'freeze', '--python', python], text=True)
    (OUT / 'environment-freeze.txt').write_text(frozen)
    child_env = dict(os.environ, GDZ_BASELINE_ENV_READY='1', PYTHONUNBUFFERED='1')
    result = subprocess.run([python, __file__], env=child_env)
    raise SystemExit(result.returncode)


def source_tree():
    archive = SCRATCH / 'multinet.tar.gz'
    download(f'https://codeload.github.com/ManifoldRG/MultiNet/tar.gz/{COMMIT}', archive)
    with tarfile.open(archive) as tar:
        tar.extractall(SCRATCH, filter='data')
    source = SCRATCH / f'MultiNet-{COMMIT}'
    root = SCRATCH / 'MultiNet'
    source.rename(root)
    return root


def minimal_prismatic_packages(vla_root):
    # Import frozen HF implementation without loading unrelated training code
    # via prismatic/__init__.py (which depends on the full training stack).
    for name, rel in [('prismatic', 'prismatic'), ('prismatic.extern', 'prismatic/extern'),
                      ('prismatic.extern.hf', 'prismatic/extern/hf')]:
        package = types.ModuleType(name)
        package.__path__ = [str(vla_root / rel)]
        sys.modules[name] = package


def prepare_runtime(kind='baseline_only'):
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(SCRATCH / 'hf')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    manifest = {'status': 'starting', 'kind': kind, 'multinet_commit': COMMIT,
                'model': MODEL, 'revision': REVISION, 'dataset': DATASET,
                'version': DATA_VERSION, 'split': SPLIT, 'seed': SEED,
                'expected_episodes': EXPECTED_EPISODES, 'expected_steps': EXPECTED_STEPS,
                'python': platform.python_version(), 'sampling': False,
                'quantization': None, 'center_crop_area': 0.9,
                'scope': 'one dataset of 20; not a full benchmark',
                'exact_paper_reproduction': False,
                'qualification': 'Kaggle runtime adaptation; compare with pinned upstream reference'}
    save_json('manifest.json', manifest)
    bootstrap()
    import numpy as np
    import torch
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

    tf.config.set_visible_devices([], 'GPU')  # TF only handles data/preprocessing.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('This inference job requires a CUDA GPU')
    # New Torch may report emulated BF16 on T4; require native Ampere+ support.
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    manifest['dtype'] = str(dtype)
    manifest['attention'] = 'eager'  # portable to T4/P100; explicitly differs from original FA2.
    manifest['gpus'] = [{'name': torch.cuda.get_device_name(i),
                         'bytes': torch.cuda.get_device_properties(i).total_memory}
                        for i in range(torch.cuda.device_count())]
    manifest['packages'] = {p: importlib.metadata.version(p) for p in
                            ('torch', 'torchvision', 'transformers', 'timm', 'tensorflow-datasets')}
    root = source_tree()
    vla_root = root / 'src/eval/profiling/openvla'
    sys.path[:0] = [str(root), str(vla_root)]
    minimal_prismatic_packages(vla_root)
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from experiments.robot.openvla_utils import crop_and_resize
    from src.eval.profiling.openvla.experiments.robot.openvla_openx_dataloader import get_openx_dataloader
    from src.eval.profiling.openvla.experiments.robot.multinet_openvla_utils import convert_action, drop_is_terminal_dim

    reference = json.loads((vla_root / 'results/openvla_openx_eval_results.json').read_text())[DATASET]
    save_json('upstream_reference_NOT_OUR_RESULT.json', reference)
    stats_path = vla_root / 'data/dataset_statistics.json'
    manifest['statistics_sha256'] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    manifest['source_sha256'] = {
        str(p.relative_to(vla_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (vla_root / 'experiments/robot').glob('*.py')}

    # Download only one official test split (about 1 GB), not its training set.
    data_dir = SCRATCH / DATASET / DATA_VERSION
    base = f'https://storage.googleapis.com/gresearch/robotics/{DATASET}/{DATA_VERSION}'
    for name in ('dataset_info.json', 'features.json'):
        download(base + '/' + name, data_dir / name)
    info = json.loads((data_dir / 'dataset_info.json').read_text())
    split = next(s for s in info['splits'] if s['name'] == SPLIT)
    if sum(map(int, split['shardLengths'])) != EXPECTED_EPISODES:
        raise ValueError('Official episode count changed')
    n_shards = len(split['shardLengths'])
    manifest['data_files'] = []
    for i in range(n_shards):
        name = f'{DATASET}-{SPLIT}.tfrecord-{i:05d}-of-{n_shards:05d}'
        target = data_dir / name
        print('Downloading', name, flush=True)
        download(base + '/' + name, target)
        manifest['data_files'].append({'name': name, 'bytes': target.stat().st_size,
                                      'sha256': hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()})
    builder = tfds.builder_from_directory(str(data_dir))
    episodes = builder.as_dataset(split=SPLIT, shuffle_files=False,
                                  read_config=tfds.ReadConfig(interleave_cycle_length=1))
    flat_dir = SCRATCH / 'flat_steps_0'
    episodes.flat_map(lambda episode: episode['steps']).save(str(flat_dir))
    save_json('manifest.json', manifest)

    print('Downloading pinned model', REVISION, flush=True)
    checkpoint = snapshot_download(MODEL, revision=REVISION,
                                   allow_patterns=['*.json', '*.model', '*.safetensors'])
    AutoConfig.register('openvla', OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    config = OpenVLAConfig.from_pretrained(checkpoint)
    # Placement may span GPUs/CPU; no 4/8-bit quantization is substituted.
    max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.85)
                  for i in range(torch.cuda.device_count())}
    max_memory['cpu'] = '20GiB'
    model = OpenVLAForActionPrediction.from_pretrained(
        checkpoint, config=config, torch_dtype=dtype, attn_implementation='eager',
        low_cpu_mem_usage=True, device_map='auto', max_memory=max_memory)
    model.norm_stats = json.loads(stats_path.read_text())
    model.eval()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=False)
    device = next(model.vision_backbone.parameters()).device
    manifest['device_map'] = {k: str(v) for k, v in model.hf_device_map.items()}
    manifest['status'] = 'evaluating'
    save_json('manifest.json', manifest)

    def prepare_inputs(image_array, instruction):
        # Matches frozen MultiNet get_vla_action, except recorded device/dtype.
        image = Image.fromarray(image_array).convert('RGB')
        tensor = tf.image.convert_image_dtype(tf.convert_to_tensor(np.array(image)), tf.float32)
        tensor = crop_and_resize(tensor, 0.9, 1)
        image = Image.fromarray(tf.image.convert_image_dtype(
            tf.clip_by_value(tensor, 0, 1), tf.uint8, saturate=True).numpy()).convert('RGB')
        prompt = f'In: What action should the robot take to {instruction.lower()}?\nOut:'
        return processor(prompt, image).to(device, dtype=dtype)

    def predict(image_array, instruction):
        inputs = prepare_inputs(image_array, instruction)
        with torch.inference_mode():
            return model.predict_action(**inputs, unnorm_key=DATASET, do_sample=False)

    return types.SimpleNamespace(
        np=np, torch=torch, model=model, manifest=manifest, reference=reference,
        flat_dir=flat_dir, get_openx_dataloader=get_openx_dataloader,
        convert_action=convert_action, drop_is_terminal_dim=drop_is_terminal_dim,
        prepare_inputs=prepare_inputs, predict=predict)


def main():
    runtime = prepare_runtime()
    np, torch = runtime.np, runtime.torch
    flat_dir, get_openx_dataloader = runtime.flat_dir, runtime.get_openx_dataloader
    convert_action, drop_is_terminal_dim = runtime.convert_action, runtime.drop_is_terminal_dim
    predict, reference, manifest = runtime.predict, runtime.reference, runtime.manifest
    errors, completed, final_matches = [], 0, 0
    start = time.monotonic()
    with (OUT / 'predictions.jsonl').open('w', buffering=1) as log:
        for episode_idx, batch in enumerate(get_openx_dataloader([str(flat_dir)], batch_size=1)):
            for step_idx in range(len(batch['continuous_observation'][0])):
                image = batch['image_observation'][0][step_idx]
                instruction = batch['text_observation'][0][step_idx]
                if image is None or not isinstance(instruction, str):
                    raise ValueError('Missing image or instruction; no silent skipping')
                torch.cuda.synchronize()
                before = time.monotonic()
                raw = predict(image, instruction)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - before
                prediction = convert_action(raw.copy(), DATASET).tolist()
                target = np.asarray(drop_is_terminal_dim(batch['action'][0][step_idx], DATASET)).tolist()
                error = mse(prediction, target)
                errors.append(error)
                last = bool(batch['is_last'][0][step_idx])
                if last:
                    completed += 1
                    final_matches += int(np.array_equal(prediction, target))
                log.write(json.dumps({'episode': episode_idx, 'step': step_idx,
                                     'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                                     'instruction': instruction, 'raw_prediction': raw.tolist(),
                                     'prediction': prediction, 'target': target, 'mse': error,
                                     'inference_seconds': elapsed, 'is_last': last}, allow_nan=False) + '\n')
            progress = {**aggregate(errors), 'status': 'partial', 'complete_episodes': completed,
                        'elapsed_seconds': time.monotonic() - start}
            save_json('progress.json', progress)
            print(json.dumps(progress), flush=True)
    result = {**aggregate(errors), 'status': 'complete', 'complete_episodes': completed,
              'elapsed_seconds': time.monotonic() - start,
              'offline_final_action_exact_match_percent': 100 * final_matches / completed,
              'peak_gpu_bytes': [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]}
    if completed != EXPECTED_EPISODES or len(errors) != EXPECTED_STEPS:
        result['status'] = 'coverage_mismatch'
        save_json('result.json', result)
        raise ValueError('Coverage differs from published baseline; inspect before claiming reproduction')
    result['delta_amse_vs_published'] = result['amse'] - reference['avg_dataset_amse']
    result['delta_namse_vs_published'] = result['namse'] - reference['normalized_amse'] if result['namse'] is not None else None
    save_json('result.json', result)
    manifest['status'] = 'complete_one_dataset'
    save_json('manifest.json', manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        save_json('failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise

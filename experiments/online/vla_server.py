"""OpenVLA action-sampling server (Hugging Face transformers backend).

Replaces RoboMonkey's SGLang engine with the reference `openvla/openvla-7b` HF
implementation so that every selection method draws from the same model code.

The upstream `generate` path rejects batch size > 1, so candidates are decoded
with an explicit loop: one multimodal prefill (vision encoded once), the KV cache
replicated to n rows, then seven batched decode steps of the Llama backbone.
temperature == 0 is greedy; otherwise ancestral sampling over the full vocabulary
(no top-k / top-p), matching SGLang's defaults used by RoboMonkey. EOS is masked so
that exactly seven tokens are produced; out-of-range tokens are returned unchanged
and filtered by the client, as upstream does. At start-up the greedy loop is checked
token-for-token against the reference `generate` call.
"""
import argparse
import threading
import time

import json_numpy as json
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ACTION_DIM = 7
EOS_ID = 2
PROMPT = 'In: What action should the robot take to {}?\nOut:'

app = FastAPI()
state = {}
lock = threading.Lock()


def load(model_id, unnorm_key, seed):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation='eager',
        low_cpu_mem_usage=True, trust_remote_code=True).to('cuda:0').eval()
    bins = np.linspace(-1, 1, model.config.n_action_bins)
    state.update(processor=processor, model=model, stats=model.norm_stats[unnorm_key]['action'],
                 bin_centers=(bins[:-1] + bins[1:]) / 2.0, vocab_size=32000,
                 generator=torch.Generator(device='cuda:0').manual_seed(seed))


def tokens_to_actions(ids):
    disc = np.clip(state['vocab_size'] - ids - 1, 0, state['bin_centers'].shape[0] - 1)
    normalized = state['bin_centers'][disc]
    q01, q99 = np.array(state['stats']['q01']), np.array(state['stats']['q99'])
    mask = np.array(state['stats'].get('mask', np.ones_like(q01, dtype=bool)))
    return np.where(mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized)


def prepare(instruction, image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = state['processor'](PROMPT.format(instruction), image).to('cuda:0', dtype=torch.bfloat16)
    input_ids = inputs['input_ids']
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], device=input_ids.device)], dim=1)
    return input_ids, inputs['pixel_values']


def _repeat_cache(cache, n):
    legacy = cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else cache
    return tuple((k.repeat_interleave(n, 0), v.repeat_interleave(n, 0)) for k, v in legacy)


def _pick(logits, temperature):
    logits = logits.float()
    logits[:, EOS_ID] = -float('inf')
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=state['generator']).squeeze(-1)


@torch.inference_mode()
def sample(instruction, image_path, n, temperature):
    model = state['model']
    input_ids, pixel_values = prepare(instruction, image_path)
    out = model(input_ids=input_ids, pixel_values=pixel_values, use_cache=True)
    cache = _repeat_cache(out.past_key_values, n)
    token = _pick(out.logits[:, -1].expand(n, -1).clone(), temperature)
    tokens = [token]
    for _ in range(ACTION_DIM - 1):
        out = model.language_model(input_ids=token[:, None], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        token = _pick(out.logits[:, -1], temperature)
        tokens.append(token)
    return torch.stack(tokens, dim=1).cpu().numpy()


@torch.inference_mode()
def check_against_reference(image_path):
    input_ids, pixel_values = prepare('pick up the object', image_path)
    reference = state['model'].generate(input_ids, pixel_values=pixel_values, max_new_tokens=ACTION_DIM,
                                        do_sample=False)[0, -ACTION_DIM:].cpu().numpy()
    ours = sample('pick up the object', image_path, 1, 0.0)[0]
    if not np.array_equal(reference, ours):
        raise RuntimeError(f'greedy loop {ours} != reference generate {reference}')


@app.post('/sample')
async def sample_endpoint(request: Request):
    body = json.loads(await request.body())
    with lock:
        torch.cuda.synchronize()
        start = time.perf_counter()
        ids = sample(body['instruction'], body['image_path'], int(body['n']), float(body['temperature']))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    return {'output_ids': ids.tolist(), 'actions': tokens_to_actions(ids).tolist(), 'seconds': seconds}


@app.get('/health')
def health():
    return {'ready': 'model' in state}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='openvla/openvla-7b')
    parser.add_argument('--unnorm-key', default='bridge_orig')
    parser.add_argument('--port', type=int, default=3200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--check-image', default=None, help='image for the start-up greedy equivalence check')
    args = parser.parse_args()
    load(args.model, args.unnorm_key, args.seed)
    if args.check_image:
        check_against_reference(args.check_image)
        print('greedy loop matches reference generate', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')

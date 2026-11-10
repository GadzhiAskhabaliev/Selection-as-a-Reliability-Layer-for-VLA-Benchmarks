#!/usr/bin/env bash
# Clones pinned upstream code and downloads checkpoints into $RM. Needs `uv` and `git`.
# The repository itself is expected at $RM/selection.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RM"
clone() {  # url dir commit
  [ -d "$2" ] || git clone --recurse-submodules "$1" "$2"
  git -C "$2" checkout -q "$3" && git -C "$2" submodule update -q --init --recursive
}
clone https://github.com/robomonkey-vla/RoboMonkey.git RoboMonkey main
clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01

[ -x envs/tools/bin/python ] || uv venv -q -p 3.10 envs/tools
VIRTUAL_ENV=$RM/envs/tools uv pip install -q "huggingface_hub[cli,hf_transfer]==0.26.2"
for repo in openvla/openvla-7b openvla/openvla-7b-finetuned-libero-10 openai/clip-vit-large-patch14-336; do
  envs/tools/bin/huggingface-cli download "$repo" --quiet >/dev/null
done
envs/tools/bin/huggingface-cli download robomonkey-vla/monkey-verifier-7b \
  --local-dir "$RM/models/monkey-verifier-7b" --quiet >/dev/null

mkdir -p stats
envs/tools/bin/python - <<'EOF'
import json, os
from huggingface_hub import hf_hub_download
for repo, key in [('openvla/openvla-7b', 'bridge_orig'), ('openvla/openvla-7b-finetuned-libero-10', 'libero_10')]:
    stats = json.load(open(hf_hub_download(repo, 'config.json')))['norm_stats'][key]['action']
    json.dump({k: stats[k] for k in ('q01', 'q99', 'mask')}, open(f"{os.environ['RM']}/stats/{key}.json", 'w'), indent=1)
EOF
echo "bootstrap done"

#!/usr/bin/env bash
# Builds one user-space environment (no sudo). Usage: setup_envs.sh {vla|libero|sim|verifier}
#   vla       OpenVLA sampling server (torch 2.2 / transformers 4.40.1)
#   libero    LIBERO-10 simulator client (robosuite 1.4.1, MuJoCo)
#   sim       SIMPLER client (SAPIEN 2.2.2; needs a GPU with Vulkan graphics support)
#   verifier  RoboMonkey verifier server, versions as in RoboMonkey scripts/env_verifier.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
venv=$RM/envs/$1
[ -x "$venv/bin/python" ] || uv venv -q -p 3.10 "$venv"
export VIRTUAL_ENV=$venv
pip_() { uv pip install "$@"; }

case "$1" in
  vla)
    pip_ "torch==2.2.0" "torchvision==0.17.0" --index-url https://download.pytorch.org/whl/cu121
    pip_ "numpy<2" "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
         "accelerate==0.30.1" sentencepiece pillow fastapi uvicorn json_numpy
    ;;
  libero)
    pip_ "torch==2.2.0" --index-url https://download.pytorch.org/whl/cpu
    pip_ "numpy==1.26.4" "tensorflow-cpu==2.15.1" "robosuite==1.4.1" "mujoco==3.1.6" "bddl==1.0.1" \
         "PyOpenGL==3.1.7" "setuptools<70" easydict cloudpickle "gym==0.25.2" "imageio[ffmpeg]" \
         pyyaml hydra-core future matplotlib requests json_numpy pillow
    pip_ --no-deps -e "$RM/LIBERO" -e "$RM/selection"
    mkdir -p "$RM/.libero"
    L=$RM/LIBERO/libero/libero
    printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
      "$L" "$L" "$L" "$L" "$L" > "$RM/.libero/config.yaml"
    ;;
  sim)
    pip_ "numpy==1.24.4" "scipy==1.12.0" "tensorflow-cpu==2.15.1" "sapien==2.2.2" "setuptools<70" \
         "gymnasium>=0.28.1,<1.0" h5py pyyaml tqdm GitPython tabulate "gdown>=4.6.0" \
         transforms3d opencv-python-headless "imageio[ffmpeg]" trimesh rtree ruckig \
         mediapy requests json_numpy "transformers==4.40.1"
    pip_ --no-deps -e "$RM/RoboMonkey/SimplerEnv/ManiSkill2_real2sim" -e "$RM/RoboMonkey/SimplerEnv" -e "$RM/selection"
    ;;
  verifier)
    pip_ -e "$RM/RoboMonkey/monkey-verifier/llava_setup/LLaVA"
    pip_ ninja
    pip_ "torch==2.0.1+cu118" "torchvision==0.15.2+cu118" "torchaudio==2.0.2" \
         --index-url https://download.pytorch.org/whl/cu118
    DS_BUILD_OPS=0 pip_ "deepspeed==0.9.3" "peft==0.4.0" "transformers==4.31.0" \
         "bitsandbytes==0.41.0" "datasets==3.6.0" "numpy==1.26.4" json_numpy openpyxl \
         fastapi uvicorn einops protobuf
    ;;
  *) echo "unknown environment $1"; exit 2 ;;
esac
"$venv/bin/python" -c "import sys; print('$1 ok', sys.version.split()[0])"

# Source before every command. All environments, caches and checkpoints live under $RM.
export RM=${RM:-$HOME/rm_bench}
export PATH=$HOME/.local/bin:$PATH
export XDG_CACHE_HOME=$RM/.cache UV_CACHE_DIR=$RM/.cache/uv PIP_CACHE_DIR=$RM/.cache/pip
export UV_HTTP_TIMEOUT=600 UV_LOCK_TIMEOUT=3600
export HF_HOME=$RM/hf TRANSFORMERS_CACHE=$RM/hf/hub HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
export LIBERO_CONFIG_PATH=$RM/.libero
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$RM/logs"

# Compute-only containers (no NVIDIA graphics capability): render on the CPU.
# MuJoCo/LIBERO works through OSMesa. SAPIEN/SIMPLER does not (see docs/ONLINE.md).
if [ "${RM_SOFTWARE_RENDERING:-0}" = 1 ]; then
  export LD_LIBRARY_PATH=$RM/sw-render/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MESA_SHADER_CACHE_DIR=$RM/.cache/mesa
  export LP_NUM_THREADS=${LP_NUM_THREADS:-8}
else
  export MUJOCO_GL=${MUJOCO_GL:-egl}
fi

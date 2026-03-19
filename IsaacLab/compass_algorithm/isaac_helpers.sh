#!/usr/bin/env bash
# Isaac helpers (old-style, deterministic) for Alvis + IsaacSim 5.1 + IsaacLab source
# Usage:
#   export IL_ROOT=.../locomotion_policy
#   export ISAACLAB_ROOT=.../IsaacLab
#   source "$IL_ROOT/isaac_helpers.sh"
#   cd "$ISAACLAB_ROOT"

# --- protect against caller's "set -e" aborting source ---
__E=0; [[ $- == *e* ]] && __E=1 && set +e

# --- sanity checks ---
if [ -z "${IL_ROOT:-}" ]; then echo "[ERROR] IL_ROOT not set"; ((__E)) && set -e; return 1; fi
if [ -z "${ISAACLAB_ROOT:-}" ]; then echo "[ERROR] ISAACLAB_ROOT not set"; ((__E)) && set -e; return 1; fi
if [ ! -d "$ISAACLAB_ROOT" ]; then echo "[ERROR] ISAACLAB_ROOT not found: $ISAACLAB_ROOT"; ((__E)) && set -e; return 1; fi

# --- container runtime ---
if command -v apptainer >/dev/null 2>&1; then
  CNT=apptainer
elif command -v singularity >/dev/null 2>&1; then
  CNT=singularity
else
  echo "[ERROR] apptainer/singularity not found"; ((__E)) && set -e; return 1
fi

# --- Isaac image ---
IMG=/apps/containers/IsaacSim/IsaacSim-NGC-5.1.0.sif
if [ ! -r "$IMG" ]; then
  echo "[ERROR] Isaac image not readable: $IMG"
  ((__E)) && set -e
  return 1
fi

# --- portable root (all caches/logs/config) ---
JOBTAG="${SLURM_JOB_ID:-manual}"
export PORTABLE_ROOT="${PORTABLE_ROOT:-$IL_ROOT/kit_portable/job_${USER}_${JOBTAG}}"
mkdir -p "$PORTABLE_ROOT"/{omni_config,xdg_cache,xdg_data,xdg_config,ov_cache,nvidia-omniverse,user_data,logs,tmp,pip_cache,torch_home,hf_home,cuda_cache} >/dev/null 2>&1 || true

# --- persistent pip base (Fixed to COMPASS_ROOT) ---
# We use COMPASS_ROOT to ensure the libraries are always found in the same folder
if [ -z "${COMPASS_ROOT:-}" ]; then
    # Fallback to current IL_ROOT if COMPASS_ROOT isn't exported yet
    PIPBASE="${IL_ROOT}/pipbase"
else
    PIPBASE="${COMPASS_ROOT}/pipbase"
fi

export PIPBASE
mkdir -p "$PIPBASE" >/dev/null 2>&1 || true

# --- optional: bind host git for GitPython ---
HOST_GIT=""
if command -v git >/dev/null 2>&1; then HOST_GIT="$(command -v git)"; fi

# --- binds (host -> container) ---
# Note: we do NOT try to override HOME (apptainer may warn). We redirect caches via XDG + binds.
BIND_ARGS=(
  --bind "$PIPBASE:/ext/pip"
  --bind "$PORTABLE_ROOT/kit_cache:/isaac-sim/kit/cache"  # FIX 1
  --bind "$PORTABLE_ROOT/kit_data:/isaac-sim/kit/data"    # FIX 2
  --bind "$PORTABLE_ROOT/kit_logs:/isaac-sim/kit/logs"    # FIX 3
  --bind "/mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_CDC/locomotion_policy/ssl11/libcrypto.so.1.1:/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1" \
  --bind "/mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_CDC/locomotion_policy/ssl11/libcrypto.so.1.1:/lib/x86_64-linux-gnu/libcrypto.so.1.1" \
  --bind "$ISAACLAB_ROOT:/workspace"
  --bind "$IL_ROOT:/project"
  --bind "$PORTABLE_ROOT:$PORTABLE_ROOT"
  --bind "$PORTABLE_ROOT/ov_cache:$HOME/.cache/ov"
  --bind "$PORTABLE_ROOT/nvidia-omniverse:$HOME/.nvidia-omniverse"
  --bind "$PORTABLE_ROOT/user_data:/isaac-sim/user_data"
)
if [ -n "$HOST_GIT" ]; then
  BIND_ARGS+=( --bind "$HOST_GIT:/usr/local/bin/git" )
fi

# --- common env for container ---
ENV_ARGS=(
  ACCEPT_EULA=Y
  PYTHONUSERBASE=/ext/pip
  PYTHONNOUSERSITE=0
  PATH=/ext/pip/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  PYTHONPATH=/ext/pip/lib/python3.11/site-packages:/workspace/source:/workspace/source/isaaclab:/workspace/source/isaaclab_tasks:/workspace/source/isaaclab_rl
  OMNI_CONFIG_PATH="$PORTABLE_ROOT/omni_config"
  XDG_CACHE_HOME="$PORTABLE_ROOT/xdg_cache"
  XDG_DATA_HOME="$PORTABLE_ROOT/xdg_data"
  XDG_CONFIG_HOME="$PORTABLE_ROOT/xdg_config"
  PIP_CACHE_DIR="$PORTABLE_ROOT/pip_cache"
  TORCH_HOME="$PORTABLE_ROOT/torch_home"
  HF_HOME="$PORTABLE_ROOT/hf_home"
  CUDA_CACHE_PATH="$PORTABLE_ROOT/cuda_cache"
  GIT_PYTHON_REFRESH=quiet
)
if [ -n "$HOST_GIT" ]; then
  ENV_ARGS+=( GIT_PYTHON_GIT_EXECUTABLE=/usr/local/bin/git )
fi

isaacenv() {
  echo "CNT=$CNT"
  echo "IMG=$IMG"
  echo "ISAACLAB_ROOT=$ISAACLAB_ROOT"
  echo "IL_ROOT=$IL_ROOT"
  echo "PORTABLE_ROOT=$PORTABLE_ROOT"
  echo "PIPBASE=$PIPBASE"
  echo "HOST_GIT=${HOST_GIT:-<none>}"
  echo "PYTHONPATH=/ext/pip/lib/python3.11/site-packages:/workspace/source:/workspace/source/isaaclab:/workspace/source/isaaclab_tasks:/workspace/source/isaaclab_rl"
}

isaacnvidia() {
  "$CNT" exec --nv --cleanenv "$IMG" nvidia-smi
}

# IMPORTANT: use IsaacSim python launcher, not python3
isaacpy() {
  "$CNT" exec --nv --cleanenv "${BIND_ARGS[@]}" --pwd /workspace "$IMG" \
    env "${ENV_ARGS[@]}" \
    /isaac-sim/python.sh "$@"
}

# pip install helper (always installs into /ext/pip via --user + PYTHONUSERBASE)
isaacpip() {
  "$CNT" exec --nv --cleanenv "${BIND_ARGS[@]}" --pwd /workspace "$IMG" \
    env "${ENV_ARGS[@]}" \
    /isaac-sim/python.sh -m pip "$@"
}

# run a repo script by relative path (like your old helper)
isaacrun() {
  local script="$1"; shift
  "$CNT" exec --nv --cleanenv "${BIND_ARGS[@]}" --pwd /workspace "$IMG" \
    env "${ENV_ARGS[@]}" \
    /isaac-sim/python.sh "/workspace/$script" "$@"
}

isaaccheck() {
  isaacpy - <<'PY'
import importlib.metadata as m
mods = ["isaaclab","isaaclab.app","isaaclab_tasks","flatdict","h5py","rsl_rl"]
for x in mods:
    __import__(x)
    print("import ok:", x)
print("rsl-rl-lib:", m.version("rsl-rl-lib"))
PY
}

export -f isaacenv isaacnvidia isaacpy isaacpip isaacrun isaaccheck
export CNT IMG PORTABLE_ROOT PIPBASE

echo "Isaac helpers ready."
echo "ISAACLAB_ROOT=$ISAACLAB_ROOT"
echo "PORTABLE_ROOT=$PORTABLE_ROOT"

((__E)) && set -e

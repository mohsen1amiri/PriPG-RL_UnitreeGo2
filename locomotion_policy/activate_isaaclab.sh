cat > "$IL_ROOT/isaac_helpers.sh" <<'SH'
#!/usr/bin/env bash
# Working-style Isaac helpers (uses /isaac-sim/python.sh)

export IL_ROOT="${IL_ROOT:?IL_ROOT not set}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:?ISAACLAB_ROOT not set}"

# container runtime
if command -v apptainer >/dev/null 2>&1; then CNT=apptainer
elif command -v singularity >/dev/null 2>&1; then CNT=singularity
else echo "[ERROR] apptainer/singularity not found"; return 1; fi

IMG=/apps/containers/IsaacSim/IsaacSim-NGC-5.1.0.sif
[ -r "$IMG" ] || { echo "[ERROR] image not readable: $IMG"; return 1; }

JOBTAG="${SLURM_JOB_ID:-manual}"
export PORTABLE_ROOT="${PORTABLE_ROOT:-$IL_ROOT/kit_portable/job_${USER}_${JOBTAG}}"
mkdir -p "$PORTABLE_ROOT"/{logs,tmp,omni_config,xdg_cache,xdg_data,xdg_config,ov_cache,nvidia-omniverse,user_data,pip_cache} >/dev/null 2>&1 || true

# persistent pip base
PIPBASE="${PIPBASE:-$IL_ROOT/pipbase}"
mkdir -p "$PIPBASE" >/dev/null 2>&1 || true

# git bind (for GitPython)
HOST_GIT=""
command -v git >/dev/null 2>&1 && HOST_GIT="$(command -v git)"

ISAAC_BIND=(
  --bind "$PIPBASE:/ext/pip"
  --bind "$ISAACLAB_ROOT:/workspace"
  --bind "$IL_ROOT:/project"
  --bind "$PORTABLE_ROOT:$PORTABLE_ROOT"
  --bind "$PORTABLE_ROOT/ov_cache:$HOME/.cache/ov"
  --bind "$PORTABLE_ROOT/nvidia-omniverse:$HOME/.nvidia-omniverse"
  --bind "$PORTABLE_ROOT/user_data:/isaac-sim/user_data"
)
[ -n "$HOST_GIT" ] && ISAAC_BIND+=( --bind "$HOST_GIT:/usr/local/bin/git" )

ISAAC_ENV=(
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
  TMPDIR="$PORTABLE_ROOT/tmp"
  GIT_PYTHON_REFRESH=quiet
)
[ -n "$HOST_GIT" ] && ISAAC_ENV+=( GIT_PYTHON_GIT_EXECUTABLE=/usr/local/bin/git )

isaacenv() {
  echo "CNT=$CNT"
  echo "IMG=$IMG"
  echo "ISAACLAB_ROOT=$ISAACLAB_ROOT"
  echo "IL_ROOT=$IL_ROOT"
  echo "PORTABLE_ROOT=$PORTABLE_ROOT"
  echo "PIPBASE=$PIPBASE"
  echo "HOST_GIT=${HOST_GIT:-<none>}"
}

isaacnvidia() { "$CNT" exec --nv --cleanenv "$IMG" nvidia-smi; }

# IMPORTANT: use IsaacSim python launcher
isaacpy() {
  "$CNT" exec --nv --cleanenv "${ISAAC_BIND[@]}" --pwd /workspace "$IMG" \
    env "${ISAAC_ENV[@]}" \
    /isaac-sim/python.sh "$@"
}

isaacpip() {
  "$CNT" exec --nv --cleanenv "${ISAAC_BIND[@]}" --pwd /workspace "$IMG" \
    env "${ISAAC_ENV[@]}" \
    /isaac-sim/python.sh -m pip install --user "$@"
}

isaacrun() {
  local script="$1"; shift
  "$CNT" exec --nv --cleanenv "${ISAAC_BIND[@]}" --pwd /workspace "$IMG" \
    env "${ISAAC_ENV[@]}" \
    /isaac-sim/python.sh "/workspace/$script" "$@"
}

isaaccheck() {
  isaacpy - <<'PY'
import importlib.metadata as m
for x in ["isaaclab","isaaclab.app","isaaclab_tasks","flatdict","h5py","rsl_rl"]:
    __import__(x)
    print("import ok:", x)
print("rsl-rl-lib:", m.version("rsl-rl-lib"))
PY
}


export -f isaacenv isaacnvidia isaacpip isaacpy isaacrun isaaccheck
echo "Isaac helpers ready. PORTABLE_ROOT=$PORTABLE_ROOT"
SH
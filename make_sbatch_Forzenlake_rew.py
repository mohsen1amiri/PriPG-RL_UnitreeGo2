#!/usr/bin/env python3
"""
SBATCH generator for go2_reap_ppo_mpc.py (UPDATED to match your *new* argparse)

Updates vs your old generator
- ExperimentDefaults now matches go2_reap_ppo_mpc.py defaults exactly (gamma=0.98, n_epochs=100, etc.)
- FrozenLake-cost defaults now match the script (goal_cost=0.0, crash_cost=200.0, action_cost=0.01)
- start_max_tries default now matches the script (200)
- run tag now also includes reward_mode (useful if you sweep it)
- safer filename formatting (handles '-' in values)
- sanity-check: grid keys must exist in go2 argparse (ARG_ORDER)

Usage:
  python3 make_sbatch.py
  cd <OUTDIR> && ./submit_all.sh
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOG = logging.getLogger("sbatch-maker")


# =============================================================================
# User configuration (edit here)
# =============================================================================

@dataclass(frozen=True)
class SlurmConfig:
    workdir: Path = Path("/mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_CDC")
    script: str = "go2_reap_ppo_mpc.py"

    account: str = "NAISS2025-22-1763"
    partition: str = "alvis"
    gpus_per_node: str = "A40:1"
    nodes: int = 1
    time_limit: str = "7-00:00:00"

    # Optional: set to None to omit
    cpus_per_task: int | None = None
    mem: str | None = None

    # Environment bootstrap
    source_line: str = "source /cephyr/users/mohsena/Alvis/portal/vscode/Paper_CDC.sh"


@dataclass(frozen=True)
class RunScheduling:
    seeds: List[int] = field(default_factory=lambda: [0])
    runs_per_file: int = 10     # MUST be divisible by len(seeds)
    max_parallel: int = 10      # python processes concurrently within one SBATCH job


@dataclass(frozen=True)
class ExperimentDefaults:
    """
    MUST match argparse names in go2_reap_ppo_mpc.py exactly.
    These defaults are updated to match your current script.
    """
    # training
    total_timesteps: int = 60_000_000
    lr: float = 1e-4
    gamma: float = 0.999
    n_steps: int = 2048
    batch_size: int = 128
    n_epochs: int = 10
    max_grad_norm: float = 1.0

    # imitation
    mse_weight: float = 1.0
    initial_beta: float = 0.0
    end_iteration_number: int = 1_000_000

    # env reward knobs (used mainly in shaped mode)
    w_dist: float = 10.0
    w_step: float = 0.5
    alive_cost: float = 1.0
    goal_reward: float = 1000.0
    crash_penalty: float = 1000.0

    w_progress: float = 100.0
    w_time: float = 1.0
    w_u: float = 0.05
    safe_margin: float = 0.15
    w_safe: float = 5.0

    # flags
    use_safety_penalty: bool = False
    always_negative_reward: bool = False
    neg_eps: float = 1e-6

    # policy network
    net_layers: int = 3
    net_nodes: int = 256

    # seed (per-run; overwritten)
    seed: int = 0

    # reward mode + FrozenLake-cost params
    reward_mode: str = "shaped"  # "shaped" or "frozenlake_cost"
    step_cost: float = 1.0
    goal_cost: float = 0.0
    crash_cost: float = 200.0
    action_cost: float = 0.01

    # evaluation args (parsed even in training)
    mode: str = "train"                   # "train" or "eval"
    eval_algo: str = "rl"                 # "rl" or "reap"
    eval_episodes: int = 20
    model_path: str = "ppo_mpc_go2_sliding_mpc_style.zip"
    deterministic: bool = False

    # during-training evaluation
    eval_freq: int = 0
    eval_n_episodes: int = 20

    # trajectories
    traj_every_train: int = 10
    traj_every_eval: int = 10
    save_eval_traj_during_train: bool = False

    # random start
    random_start: bool = False
    start_clearance: float = 0.25
    start_border_clearance: float = 0.15
    start_max_tries: int = 200

    # crash handling
    continue_on_crash: bool = False

    # logging
    log_root: str = "tb_logs"


@dataclass(frozen=True)
class SweepConfig:
    """
    Grid parameters. Add/remove entries freely.
    IMPORTANT: keys must match argparse names in go2_reap_ppo_mpc.py.
    """
    grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        # RL hyperparams
        "lr": [1e-4],
        "n_steps": [1024, 2048],
        "batch_size": [128],
        "net_layers": [3],
        "net_nodes": [256, 512],

        # toggles you said you want to sweep
        "random_start": [False, True],
        "continue_on_crash": [True],

        # example: if you ever want to sweep reward mode:
        "reward_mode": ["shaped"],
    })


# =============================================================================
# Internal helpers
# =============================================================================

# store_true flags in go2_reap_ppo_mpc.py
BOOL_FLAGS = {
    "use_safety_penalty",
    "always_negative_reward",
    "deterministic",
    "save_eval_traj_during_train",
    "random_start",
    "continue_on_crash",
}

# Mirrors go2_reap_ppo_mpc.py parse_args order
ARG_ORDER = [
    # training
    "total_timesteps", "lr", "gamma", "n_steps", "batch_size", "n_epochs", "max_grad_norm",
    # imitation
    "mse_weight", "initial_beta", "end_iteration_number",
    # env reward knobs
    "w_dist", "w_step", "alive_cost", "goal_reward", "crash_penalty",
    "w_progress", "w_time", "w_u", "safe_margin", "w_safe",
    "use_safety_penalty", "always_negative_reward", "neg_eps",
    # policy
    "net_layers", "net_nodes", "seed",
    # reward mode / frozenlake_cost
    "reward_mode", "step_cost", "goal_cost", "crash_cost", "action_cost",
    # eval
    "mode", "eval_algo", "eval_episodes", "model_path", "deterministic",
    "eval_freq", "eval_n_episodes",
    # trajectories
    "traj_every_train", "traj_every_eval", "save_eval_traj_during_train",
    # random start
    "random_start", "start_clearance", "start_border_clearance", "start_max_tries",
    # crash handling
    "continue_on_crash",
    # logging
    "log_root",
]

# Per-run log tag keys (include reward_mode now)
LOG_TAG_KEYS = [
    "reward_mode",
    "seed", "lr", "gamma", "n_steps", "batch_size",
    "net_layers", "net_nodes",
    "random_start", "continue_on_crash",
    "step_cost", "goal_cost", "crash_cost", "action_cost",
]

def _fmt_val(v: Any) -> str:
    """Filename-safe compact formatting."""
    if isinstance(v, bool):
        s = "1" if v else "0"
    elif isinstance(v, float):
        s = f"{v:g}"
    else:
        s = str(v)

    # keep filenames safe-ish
    s = s.replace("+", "")
    s = s.replace("-", "m")    # minus -> m  (e.g., -0.1 -> m0p1)
    s = s.replace(".", "p")    # dot -> p
    s = s.replace("/", "_")
    s = s.replace(":", "_")
    return s

def make_run_tag(argsd: Dict[str, Any]) -> str:
    short = {
        "reward_mode": "rm",
        "seed": "s",
        "lr": "lr",
        "gamma": "g",
        "n_steps": "ns",
        "batch_size": "bs",
        "net_layers": "L",
        "net_nodes": "H",
        "random_start": "rs",
        "continue_on_crash": "coc",
        "step_cost": "sc",
        "goal_cost": "gc",
        "crash_cost": "cc",
        "action_cost": "ac",
    }
    parts: List[str] = []
    for k in LOG_TAG_KEYS:
        if k in argsd:
            parts.append(f"{short.get(k, k)}{_fmt_val(argsd[k])}")
    return "_".join(parts)

def make_outdir_name(reward_mode: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sbatch_{reward_mode}_grid_{ts}"

def build_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]

def validate_schedule(seeds: List[int], runs_per_file: int) -> None:
    if runs_per_file <= 0:
        raise ValueError("runs_per_file must be > 0")
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")
    if runs_per_file % len(seeds) != 0:
        raise ValueError(
            f"runs_per_file={runs_per_file} must be divisible by len(seeds)={len(seeds)}"
        )

def validate_grid_keys(grid: Dict[str, List[Any]]) -> None:
    allowed = set(ARG_ORDER)
    bad = [k for k in grid.keys() if k not in allowed]
    if bad:
        raise ValueError(
            "Grid contains keys not in go2 argparse/ARG_ORDER:\n  "
            + ", ".join(bad)
            + "\nFix the key names or add them to ARG_ORDER if you added new argparse options."
        )

def cli_from_args(script: str, args: Dict[str, Any]) -> List[str]:
    """
    Build: ["isaacpy", script, "--k", "v", "--flag", ...]
    store_true flags are included only if True.
    """
    cmd = ["isaacpy", script]

    for k in ARG_ORDER:
        if k not in args:
            continue
        v = args[k]
        if v is None:
            continue

        flag = f"--{k}"
        if k in BOOL_FLAGS:
            if bool(v):
                cmd.append(flag)
            continue

        cmd.extend([flag, str(v)])

    return cmd

def shell_quote(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)

def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]

def write_sbatch(
    path: Path,
    jobname: str,
    slurm: SlurmConfig,
    scheduling: RunScheduling,
    runs: List[Tuple[Dict[str, Any], List[str]]],  # (argsdict, cmdlist)
    logdir: Path,
) -> None:
    """
    Writes one SBATCH script that executes 'runs' with internal parallelism.
    Creates per-run logs whose names encode run parameters.
    """
    out_path = logdir / f"{jobname}_%j.out"
    err_path = logdir / f"{jobname}_%j.err"

    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("#SBATCH --export=ALL\n")
        f.write(f"#SBATCH -A {slurm.account}\n")
        f.write(f"#SBATCH -p {slurm.partition}\n")
        f.write(f"#SBATCH -N {slurm.nodes}\n")
        f.write(f"#SBATCH --gpus-per-node={slurm.gpus_per_node}\n")
        f.write(f"#SBATCH -t {slurm.time_limit}\n")
        f.write(f"#SBATCH -J {jobname}\n")
        f.write(f"#SBATCH -o {out_path}\n")
        f.write(f"#SBATCH -e {err_path}\n")
        if slurm.cpus_per_task is not None:
            f.write(f"#SBATCH --cpus-per-task={slurm.cpus_per_task}\n")
        if slurm.mem is not None:
            f.write(f"#SBATCH --mem={slurm.mem}\n")
        f.write("\n")

        f.write("set -euo pipefail\n")
        f.write("umask 027\n\n")

        f.write("echo \"Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}\"\n")
        f.write("echo \"Node(s): ${SLURM_JOB_NODELIST}\"\n")
        f.write("echo \"Start: $(date -Is)\"\n\n")

        f.write(slurm.source_line + "\n")
        f.write(f"cd {shlex.quote(str(slurm.workdir))}\n")
        f.write("echo \"Running in: $(pwd)\"\n\n")

        f.write(f"LOGDIR={shlex.quote(str(logdir))}\n")
        f.write("mkdir -p \"$LOGDIR\"\n\n")

        f.write(f"MAX_PARALLEL={int(scheduling.max_parallel)}\n")
        f.write("pids=()\n")
        f.write("cleanup() {\n")
        f.write("  for pid in \"${pids[@]:-}\"; do\n")
        f.write("    kill \"$pid\" 2>/dev/null || true\n")
        f.write("  done\n")
        f.write("}\n")
        f.write("trap cleanup EXIT INT TERM\n\n")

        f.write("run_bg() {\n")
        f.write("  local OUT_FILE=\"$1\"\n")
        f.write("  local ERR_FILE=\"$2\"\n")
        f.write("  shift 2\n")
        f.write("  \"$@\" >\"$OUT_FILE\" 2>\"$ERR_FILE\" &\n")
        f.write("  pids+=(\"$!\")\n")
        f.write("  if [ ${#pids[@]} -ge \"$MAX_PARALLEL\" ]; then\n")
        f.write("    echo \">> wait (batch of $MAX_PARALLEL)\"\n")
        f.write("    for pid in \"${pids[@]}\"; do\n")
        f.write("      wait \"$pid\" || exit 1\n")
        f.write("    done\n")
        f.write("    pids=()\n")
        f.write("  fi\n")
        f.write("}\n\n")

        for argsd, cmd in runs:
            run_tag = make_run_tag(argsd)
            f.write(f"\ntag={shlex.quote(run_tag)}\n")
            f.write("OUT_FILE=\"$LOGDIR/${tag}_${SLURM_JOB_ID}.out\"\n")
            f.write("ERR_FILE=\"$LOGDIR/${tag}_${SLURM_JOB_ID}.err\"\n")
            f.write("echo \"RUN: ${tag}\"\n")
            f.write("run_bg \"$OUT_FILE\" \"$ERR_FILE\" " + shell_quote(cmd) + "\n")

        f.write("\necho \">> final wait\"\n")
        f.write("for pid in \"${pids[@]:-}\"; do wait \"$pid\" || exit 1; done\n")
        f.write("echo \"All runs finished.\"\n")
        f.write("echo \"End: $(date -Is)\"\n")

    os.chmod(path, 0o755)

def write_submit_all(outdir: Path) -> None:
    submit_path = outdir / "submit_all.sh"
    with open(submit_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write("shopt -s nullglob\n")
        f.write("for s in *; do\n")
        f.write("  [[ \"$s\" =~ ^[0-9]+$ ]] || continue\n")
        f.write("  echo \"Submitting $s\"\n")
        f.write("  sbatch \"$s\"\n")
        f.write("done\n")
    os.chmod(submit_path, 0o755)

def write_manifest(outdir: Path, all_runs: List[Dict[str, Any]]) -> None:
    manifest_jsonl = outdir / "runs_manifest.jsonl"
    with open(manifest_jsonl, "w") as f:
        for r in all_runs:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    manifest_summary = outdir / "runs_summary.json"
    summary = {"n_runs": len(all_runs), "created_at": datetime.now().isoformat()}
    with open(manifest_summary, "w") as f:
        json.dump(summary, f, indent=2)

    LOG.info("Wrote manifest: %s", manifest_jsonl)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    slurm = SlurmConfig()
    scheduling = RunScheduling()
    defaults = ExperimentDefaults()
    sweep = SweepConfig()

    validate_schedule(scheduling.seeds, scheduling.runs_per_file)
    validate_grid_keys(sweep.grid)

    outdir = Path(make_outdir_name(defaults.reward_mode)).resolve()
    logdir = outdir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    grid_cfgs = build_grid(sweep.grid)
    configs_per_file = scheduling.runs_per_file // len(scheduling.seeds)
    chunks = chunk_list(grid_cfgs, configs_per_file)

    LOG.info("OUTDIR: %s", outdir)
    LOG.info("WORKDIR: %s", slurm.workdir)
    LOG.info("Total GRID configs: %d", len(grid_cfgs))
    LOG.info("Seeds: %d", len(scheduling.seeds))
    LOG.info("Runs per sbatch: %d (%d configs × %d seeds)",
             scheduling.runs_per_file, configs_per_file, len(scheduling.seeds))
    LOG.info("Total runs: %d", len(grid_cfgs) * len(scheduling.seeds))
    LOG.info("SBATCH files: %d", len(chunks))
    LOG.info("Internal parallelism per sbatch: %d", scheduling.max_parallel)
    LOG.info("Sweeping keys: %s", ", ".join(sweep.grid.keys()))

    defaults_dict = asdict(defaults)
    all_runs_manifest: List[Dict[str, Any]] = []

    for j, cfg_chunk in enumerate(chunks):
        jobname = f"{defaults.reward_mode}_{j:04d}"
        sbatch_path = outdir / f"{j}"

        runs_for_this_job: List[Tuple[Dict[str, Any], List[str]]] = []

        for cfg in cfg_chunk:
            for seed in scheduling.seeds:
                argsd = dict(defaults_dict)
                argsd.update(cfg)
                argsd["seed"] = int(seed)

                cmd = cli_from_args(slurm.script, argsd)
                runs_for_this_job.append((argsd, cmd))
                all_runs_manifest.append(argsd)

        write_sbatch(
            path=sbatch_path,
            jobname=jobname,
            slurm=slurm,
            scheduling=scheduling,
            runs=runs_for_this_job,
            logdir=logdir,
        )

    write_submit_all(outdir)
    write_manifest(outdir, all_runs_manifest)

    LOG.info("Done.")
    LOG.info("Created sbatch files in: %s", outdir)
    LOG.info("Submit with: cd %s && ./submit_all.sh", outdir)
    LOG.info("Per-run logs will be in: %s", logdir)


if __name__ == "__main__":
    main()

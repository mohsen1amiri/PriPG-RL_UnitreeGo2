#!/usr/bin/env python3
import os
from itertools import product

# -----------------------------
# USER SETTINGS
# -----------------------------
OUTDIR = "sbatch_frozenlake_many"
WORKDIR = "/mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_CDC"
SCRIPT = "go2_reap_ppo_mpc.py"

ACCOUNT = "NAISS2025-22-1763"
PARTITION = "alvis"
GPUS = "A40:1"
NODES = 1
TIME = "7-00:00:00"

SOURCE_LINE = "source /cephyr/users/mohsena/Alvis/portal/vscode/Paper_CDC.sh"

SEEDS = [0, 1, 2, 3, 4]
RUNS_PER_FILE = 10  # configs_per_file * len(SEEDS)
MAX_PARALLEL = 10   # run 10 at once on the same GPU

# -----------------------------
# NEW (matches updated go2_reap_ppo_mpc.py): trajectory saving knobs
# -----------------------------
TRAJ_EVERY_TRAIN = 10          # 0 disables train trajectory PNGs
TRAJ_EVERY_EVAL = 10           # 0 disables eval trajectory PNGs (for eval mode + periodic eval)
SAVE_EVAL_TRAJ_DURING_TRAIN = True  # if True, periodic eval during training will also save PNGs

# -----------------------------
# During-training evaluation knobs
# -----------------------------
EVAL_FREQ = 100_000        # 0 disables periodic eval during training
EVAL_N_EPISODES = 20       # number of episodes per periodic eval
EVAL_DETERMINISTIC = True  # adds --deterministic

# -----------------------------
# Safety penalty handling (IMPORTANT)
# In your env __init__ you set use_safety_penalty=True,
# BUT main() overwrites it with args.use_safety_penalty (default False)
# unless you pass the flag.
# -----------------------------
USE_SAFETY_PENALTY = False  # set False if you explicitly want safety penalty OFF

# -----------------------------
# Fixed FrozenLake-cost reward params (keep constant)
# -----------------------------
REWARD_ARGS = [
    "--reward_mode", "frozenlake_cost",
    "--step_cost", "1.0",
    "--goal_cost", "0.0",
    "--crash_cost", "200.0",
    "--action_cost", "0.00",
]

# -----------------------------
# Other fixed knobs
# -----------------------------
FIXED_ARGS = [
    "--gamma", "0.98",
    "--total_timesteps", "5_000_000",
    "--n_epochs", "10",
    "--max_grad_norm", "10.0",
    "--mse_weight", "1.0",
    "--initial_beta", "1.0",
    "--safe_margin", "0.15",
    "--w_safe", "5",
    "--log_root", "tb_logs_frozenlake",

    # NEW: trajectory flags in updated training script
    "--traj_every_train", str(TRAJ_EVERY_TRAIN),
    "--traj_every_eval", str(TRAJ_EVERY_EVAL),
]

if USE_SAFETY_PENALTY:
    FIXED_ARGS.append("--use_safety_penalty")

if SAVE_EVAL_TRAJ_DURING_TRAIN:
    FIXED_ARGS.append("--save_eval_traj_during_train")

# -----------------------------
# Eval args (only used if EVAL_FREQ > 0)
# -----------------------------
EVAL_ARGS = [
    "--eval_freq", str(EVAL_FREQ),
    "--eval_n_episodes", str(EVAL_N_EPISODES),
]
if EVAL_DETERMINISTIC:
    EVAL_ARGS.append("--deterministic")

# Grid
GRID = {
    "lr": ["1e-4"],
    "n_steps": ["1024", "2048"],
    "batch_size": ["128"],
    "net_layers": ["2", "3"],
    "net_nodes": ["256", "512"],
    "end_iteration_number": ["300_000"],
}

# -----------------------------
# GENERATION
# -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    logdir = os.path.abspath(os.path.join(OUTDIR, "logs"))
    os.makedirs(logdir, exist_ok=True)

    if RUNS_PER_FILE % len(SEEDS) != 0:
        raise ValueError(
            f"RUNS_PER_FILE={RUNS_PER_FILE} must be divisible by number of seeds={len(SEEDS)}"
        )
    configs_per_file = RUNS_PER_FILE // len(SEEDS)

    keys = list(GRID.keys())
    all_cfgs = [dict(zip(keys, values)) for values in product(*[GRID[k] for k in keys])]
    chunks = [all_cfgs[i:i + configs_per_file] for i in range(0, len(all_cfgs), configs_per_file)]

    print(f"OUTDIR: {OUTDIR}")
    print(f"WORKDIR: {WORKDIR}")
    print(f"Total configs: {len(all_cfgs)}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Runs per file: {RUNS_PER_FILE} = {configs_per_file} configs × {len(SEEDS)} seeds")
    print(f"Total runs: {len(all_cfgs) * len(SEEDS)}")
    print(f"SBATCH files: {len(chunks)}")
    print(f"Parallelism per sbatch: {MAX_PARALLEL}")
    print(f"Eval during train: eval_freq={EVAL_FREQ} eval_n_episodes={EVAL_N_EPISODES} det={EVAL_DETERMINISTIC}")
    print(f"Traj: train_every={TRAJ_EVERY_TRAIN} eval_every={TRAJ_EVERY_EVAL} save_eval_traj_during_train={SAVE_EVAL_TRAJ_DURING_TRAIN}")
    print(f"Safety penalty enabled: {USE_SAFETY_PENALTY}")

    for j, cfg_chunk in enumerate(chunks):
        jobname = f"flk_{j:04d}"
        sbatch_path = os.path.join(OUTDIR, str(j))  # numeric filename only

        with open(sbatch_path, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f"#SBATCH -A {ACCOUNT} -p {PARTITION}\n")
            f.write(f"#SBATCH -N {NODES} --gpus-per-node={GPUS}\n")
            f.write(f"#SBATCH -t {TIME}\n")
            f.write(f"#SBATCH -J {jobname}\n")
            f.write(f"#SBATCH -o {logdir}/{jobname}_%j.out\n")
            f.write(f"#SBATCH -e {logdir}/{jobname}_%j.err\n\n")

            f.write("set -e\n")
            f.write(SOURCE_LINE + "\n\n")
            f.write(f'cd "{WORKDIR}"\n\n')

            f.write('echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"\n')
            f.write('echo "Running in: $(pwd)"\n')
            f.write(f"MAX_PARALLEL={MAX_PARALLEL}\n\n")

            # Robust parallel runner
            f.write("pids=()\n")
            f.write("run_bg() {\n")
            f.write("  \"$@\" &\n")
            f.write("  pids+=(\"$!\")\n")
            f.write("  if [ ${#pids[@]} -ge \"$MAX_PARALLEL\" ]; then\n")
            f.write("    echo \">> wait (batch of $MAX_PARALLEL)\"\n")
            f.write("    for pid in \"${pids[@]}\"; do\n")
            f.write("      wait \"$pid\" || exit 1\n")
            f.write("    done\n")
            f.write("    pids=()\n")
            f.write("  fi\n")
            f.write("}\n\n")

            for cfg in cfg_chunk:
                for seed in SEEDS:
                    cmd = [
                        "isaacpy", SCRIPT,
                        "--seed", str(seed),
                        "--lr", cfg["lr"],
                        "--n_steps", cfg["n_steps"],
                        "--batch_size", cfg["batch_size"],
                        "--net_layers", cfg["net_layers"],
                        "--net_nodes", cfg["net_nodes"],
                        "--end_iteration_number", cfg["end_iteration_number"],
                    ] + REWARD_ARGS + FIXED_ARGS

                    if EVAL_FREQ > 0:
                        cmd += EVAL_ARGS
                    else:
                        cmd += ["--eval_freq", "0"]

                    f.write("\n")
                    f.write(
                        'echo "RUN: '
                        f'seed={seed} lr={cfg["lr"]} n_steps={cfg["n_steps"]} bs={cfg["batch_size"]} '
                        f'L={cfg["net_layers"]} H={cfg["net_nodes"]} end={cfg["end_iteration_number"]} '
                        f'eval_freq={EVAL_FREQ} eval_eps={EVAL_N_EPISODES} det={int(EVAL_DETERMINISTIC)} '
                        f'traj_train={TRAJ_EVERY_TRAIN} traj_eval={TRAJ_EVERY_EVAL} save_eval_traj={int(SAVE_EVAL_TRAJ_DURING_TRAIN)} '
                        f'safety={int(USE_SAFETY_PENALTY)}"\n'
                    )
                    f.write("run_bg " + " ".join(cmd) + "\n")

            f.write("\necho \">> final wait\"\n")
            f.write("for pid in \"${pids[@]}\"; do wait \"$pid\" || exit 1; done\n")
            f.write("echo \"All runs finished.\"\n")

        os.chmod(sbatch_path, 0o755)

    submit_path = os.path.join(OUTDIR, "submit_all.sh")
    with open(submit_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n")
        f.write("shopt -s nullglob\n")
        f.write("for s in *; do\n")
        f.write("  [[ \"$s\" =~ ^[0-9]+$ ]] || continue\n")
        f.write("  echo \"Submitting $s\"\n")
        f.write("  sbatch \"$s\"\n")
        f.write("done\n")
    os.chmod(submit_path, 0o755)

    print(f"\nDone.\nCreated sbatch files in: {OUTDIR}\nSubmit with:\n  cd {OUTDIR} && ./submit_all.sh\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import itertools
import math
import json   # ← we’ll need this if you want to pass dicts


root_path = './sbatch_v1/'

project_ids = [
    'naiss2024-22-1645',
    'naiss2024-22-1289'
]
py_cmd = 'python PPO_v1.py'

# Parameters as per argparse
seeds = [1357924680, 2468135790, 3141592653, 2718281828, 1618033988]
MSE_weights = [1]
end_iterations = [320_400] # 400 episodes
total_timesteps_list = [4_806_000] # 6000 episodes
n_epochs_list = [10]
start_iterations = [160_200] # 200 episodes
gammas = [0.98]
initial_betas = [1.0]
n_steps_list = [801]
mpc_error_ratios = [0.0, 0.01, 0.1, 0.5, 0.9]
# mpc_error_ratios = [0.0]
# MPC_senario = ["biased_MPC", "noisey_MPC"]
MPC_senario = ["biased_MPC"]
safety_const = ["True"]
margin = [3]
naive_reward = ["False"]


# ---------- NEW parameter grids ----------
# learning_rates         = [1e-4, 1e-3]                  # <-- example values
learning_rates         = [1e-4]
# target_kls             = [0.2, 0.03, None]                  # <-- None will print as "None"
# target_kls             = [0.2, 0.03] 
target_kls             = [0.2] 
# ent_coefs              = [0.0, 0.001]                   # <-- entropy coeffs to sweep
ent_coefs              = [0.0] 
# clip_range_types       = ["Dec", "Inc", "fixed"]        # <-- just a string flag
# clip_range_types       = ["Dec", "Inc"]
clip_range_types       = ["Inc"]
policy_kwargs_list     = [
    20
    # json.dumps({"net_arch":[128,128]})
]  # <<– each entry becomes: --policy_kwargs '{"net_arch":[20,20]}'


# Generate all command strings
param_lists = [
    ('seed', seeds),
    ('w', MSE_weights),
    ('e', end_iterations),
    ('total_timesteps', total_timesteps_list),
    ('n_epochs', n_epochs_list),
    ('start_iteration_number', start_iterations),
    ('gamma', gammas),
    ('Initial_Beta', initial_betas),
    ('n_steps', n_steps_list),
    ('MPC_error_ratio', mpc_error_ratios),
    ("MPC_senario", MPC_senario),
    ("safety_const", safety_const),
     # ── here are the new ones: ───────────────────────────────
    ('learning_rate',            learning_rates),
    ('target_kl',                target_kls),
    ('ent_coef',                 ent_coefs),
    ('clip_range_type',          clip_range_types),
    ('policy_net_arch',            policy_kwargs_list),
    ("naive_reward",                naive_reward),
    ("margin",                      margin)
]
keys, values = zip(*param_lists)
all_cmds = [
    ' '.join([py_cmd] + [f"--{k} {v}" for k, v in zip(keys, combo)])
    for combo in itertools.product(*values)
]

# Prepare batching: four commands per script
num_scripts = math.ceil(len(all_cmds) / 4)
half = num_scripts // 2

os.makedirs(root_path, exist_ok=True)
# i0 = 0
i0 = 52
for idx in range(0, len(all_cmds), 4):
    i0 += 1
    # select project id: first half uses first, second half uses second
    proj = project_ids[0] if i0 <= i0 + half else project_ids[1]
    # prepare SBATCH header and environment lines dynamically
    lines = [
        '#!/usr/bin/env bash',
        f'#SBATCH -A {proj} -p alvis',
        '#SBATCH -N 1 --gpus-per-node=A40:1',
        '#SBATCH -t 2-00:00:00',
        'ml load CUDA-Python/12.1.0-gfbf-2023a-CUDA-12.1.1',
        'source /mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_KTH_Hitachi/RL_Voltage_venv_v1/bin/activate',
        'cd /mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_KTH_Hitachi'
    ]
    # combine up to four commands and add wait at end
    cmd_list = all_cmds[idx:idx+4]
    combined_cmd = ' & '.join(cmd_list) + ' & wait'

    # write sbatch file
    filepath = os.path.join(root_path, str(i0))
    with open(filepath, 'w') as f:
        for line in lines:
            f.write(line + '\n')
        f.write(combined_cmd + '\n')
    print(f"Generated #{i0} (project {proj}): {combined_cmd}")

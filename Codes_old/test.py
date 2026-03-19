import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import gymnasium       # or `import gym` if you registered VSC_Env under gym
from stable_baselines3 import PPO
import Custom_env.gymnasium_env
import csv  # at the top of your script





results_summary = []  # List to store result rows for CSV


def evaluate_mpc(env, n_episodes=2, gamma=0.98, max_steps=None, trim_lower=50):
    episode_stats = []
    discounted_returns = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False

        # per-episode accumulators
        real_vals, imag_vals = [], []
        discounted_return = 0.0
        discount = 1.0
        steps = 0

        while not done:
            action = info["Action MPC"]
            obs, reward, done, trunc, info = env.step(action)

            # record measurements
            i_real, i_imag = obs["I_meas"]
            real_vals.append(i_real)
            imag_vals.append(i_imag)

            # accumulate discounted return
            discounted_return += discount * reward
            discount *= gamma

            steps += 1
            if max_steps and steps >= max_steps:
                break

        # collect episode data
        discounted_returns.append(discounted_return)
        violation_count = info["constraint violation number"]
        va = info["Voltage_reconstructed_a"][trim_lower:]
        vb = info["Voltage_reconstructed_b"][trim_lower:]
        vc = info["Voltage_reconstructed_c"][trim_lower:]

        episode_stats.append({
            "episode": ep,
            "discounted_return": discounted_return,
            "violations": violation_count,
            "i_real": real_vals,
            "i_imag": imag_vals,
            "va": va,
            "vb": vb,
            "vc": vc,
            "steps": steps
        })

        print(f"MPC: Episode {ep:>2d} | return = {discounted_return:8.2f} "
              f"| violations = {violation_count:3d} | steps = {steps}")

    # overall average
    avg_return = np.mean(discounted_returns)

    return {
        "episode_stats": episode_stats,
        "average_return": avg_return
    }



def inject_clipped_variance(param,
                                var_ratio,
                                min_frac=1e-12,
                                max_frac=2.0):
        """
        param           : nominal positive value
        var_ratio       : noise variance relative to param^2 (e.g. 0.10 for “10% variance”)
        min_frac        : lower bound as a fraction of param (default → non‐negative)
        max_frac        : upper bound as a fraction of param (default → up to 200% of nominal)
        
        Returns:
        noisy_clipped : the parameter after Gaussian noise and clipping
        percent_error : (noisy_clipped - param) / param * 100
        """
        # std = np.sqrt(var_ratio) * param
        # noisy = param + std * np.random.randn()
        # low  = min_frac * param
        # high = max_frac * param if max_frac is not None else None

        # noisy_clipped = np.clip(noisy, low, high)
        # error         = noisy_clipped - param

        error = var_ratio * param

        noisy_clipped = param + error

        percent_error = round((error / param) * 100.0, 2)

        return noisy_clipped, percent_error


Rf=0.013
Lf=2.5e-3
C=30e-6
Rl=50

algo = 'PPO_MPC'
envname = 'VSC_Env-v0'
MPC_senario = "biased_MPC"



# Base directories
base_address = "/mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_KTH_Hitachi/Results_VSC_Env-v0"
plots_base   = os.path.join(base_address, "test_results")

# Evaluation parameters
gamma        = 0.98
num_episodes = 1

dt    = 1e-6
t_max = 0.04 * 2
trim_lower = 0
t     = np.arange(0, t_max + dt, dt)[trim_lower:]

Baseline_runs = [
            "VSC_Env-v0_PPO_MPC_Safety_False_only_RL__W_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_True_only_RL__W_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_True_NRew_True_only_RL__W_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"
            ]

Best_runs = [
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.01_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.01_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.1_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.1_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.5_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.5_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.9_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.9_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20",
            ]

Best_seeds = {
            "VSC_Env-v0_PPO_MPC_Safety_False_only_RL__W_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"          : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_True_only_RL__W_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"           : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_True_NRew_True_only_RL__W_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20" : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"          : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.0_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"           : ["2468135790"],
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.01_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"         : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.01_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"          : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.1_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"          : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.1_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"           : ["1618033988"],
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.5_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"          : ["2468135790"],
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.5_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"           : ["1357924680"],
            "VSC_Env-v0_PPO_MPC_Safety_False_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.9_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"          : ["2718281828"],
            "VSC_Env-v0_PPO_MPC_Safety_True_W_1.0_Beta_1.0_s_160200_e_320400_gma_0.98_TStp_4806000_NStp_801_NEpc_400_er_0.9_Sen_biased_MPC_lr_0.0001_kl_0.2_entc_0.0_clpr_Inc_arch_20-20"           : ["2468135790"]
            }

# THD computation
def compute_thd(signal, fs, f0=50):
    # print(f"max of signal: {np.max(signal)}")
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(N, 1/fs)
    # fundamental amplitude
    idx1 = np.argmin(np.abs(freqs - f0))
    V1 = np.abs(fft_vals[idx1])
    # all higher harmonics
    max_harm = int((fs/2)//f0)
    harm_idxs = [np.argmin(np.abs(freqs - n*f0)) for n in range(2, max_harm+1)]
    Vh2 = sum(np.abs(fft_vals[i])**2 for i in harm_idxs)
    return np.sqrt(Vh2) / V1 * 100

# # Loop through all "seeds" folders and evaluate models
# for folder_name in os.listdir(base_address):
#     if "seeds" not in folder_name:
#         continue
#     seed_dir = os.path.join(base_address, folder_name)
#     if not os.path.isdir(seed_dir):
#         continue
version_groups = {}
for runs in [Best_runs, Baseline_runs]:
    for run in runs:
        for folder_name in os.listdir(base_address):
            if run in folder_name and "_sd_" in folder_name:
                seed_dir = os.path.join(base_address, folder_name)
                if not os.path.isdir(seed_dir):
                    continue
                version_groups.setdefault(run, []).append(seed_dir)

# print(version_groups)
    

for run, seed_dir in version_groups.items():
    # Create corresponding plots directory
    plot_dir = os.path.join(plots_base, run)
    os.makedirs(plot_dir, exist_ok=True)

    if "Safety_True" in run:
        safety_const = True
    elif "Safety_False" in run:
        safety_const = False


    err = float(run.split("_er_")[1].split("_")[0])
    

    MPC_error_ratio = err
    var_ratio = MPC_error_ratio            # “10% error” refers to variance = 0.1·param²
    Rf_MPC, Rf_MPC_percent_error = inject_clipped_variance(Rf,  var_ratio) 
    Lf_MPC, Lf_MPC_percent_error = inject_clipped_variance(Lf,  var_ratio) 
    C_MPC, C_MPC_percent_error  = inject_clipped_variance(C,   var_ratio) 
    Rl_MPC, Rl_MPC_percent_error = inject_clipped_variance(Rl,  var_ratio) 

    env = gymnasium.make(
                envname,
                MPC_error_ratio=MPC_error_ratio,
                senario=MPC_senario,
                action_discrete=True,
                Rf_MPC=Rf_MPC,
                Lf_MPC=Lf_MPC,
                C_MPC=C_MPC,
                Rload_MPC=Rl_MPC,
                t_max=t_max,
                safety_const = safety_const,
                MPC_mode=False,
                test_mode=True
            )

    env_MPC = gymnasium.make(
                envname,
                MPC_error_ratio=MPC_error_ratio,
                senario=MPC_senario,
                action_discrete=True,
                Rf_MPC=Rf_MPC,
                Lf_MPC=Lf_MPC,
                C_MPC=C_MPC,
                Rload_MPC=Rl_MPC,
                t_max=t_max,
                safety_const = False,
                MPC_mode=True,
                test_mode=True
            )

    # run evaluation
    results = evaluate_mpc(env_MPC, n_episodes=num_episodes, gamma=gamma, trim_lower=trim_lower)

    # unpack
    episode_stats = results["episode_stats"]
    avg_return   = results["average_return"]

    # summary stats
    returns    = [r["discounted_return"] for r in episode_stats]
    violations = [r["violations"]       for r in episode_stats]

    # grab the first episode’s stats
    ep1 = results["episode_stats"][0]   # index 0 == episode 1

    # now extract what you need
    i_real = ep1["i_real"]
    i_imag = ep1["i_imag"]
    va     = ep1["va"]
    vb     = ep1["vb"]
    vc     = ep1["vc"]


    # Plot I_meas distribution
    # ——— build a 0→1 “time” array over the full history ———
    N = len(i_real)
    time_idx = np.arange(N)

    fig, ax = plt.subplots()
    # draw circle first (so it sits behind)
    circle = plt.Circle((0, 0), radius=30, edgecolor='red', fill=False, linewidth=2)
    ax.add_patch(circle)

    # scatter with clean args
    norm = Normalize(vmin=0, vmax=N-1)
    sc = ax.scatter(
                i_real, i_imag,
                marker='o',
                c=time_idx,
                cmap='viridis',
                norm=norm,
                s=10,            # bump size up a bit
                linewidths=0.5,  # thin edges
                alpha=0.8
            )

    # labels, aspect, title
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    # ax.set_title(
    #     f"The Measured Current Distribution\n"
    #     f"(err: {err}, safety: {safety_const}, sd: {sd_value}, r: {total_disc:.2f})"
    # )

    # colorbar
    cbar = fig.colorbar(sc, ax=ax, label="Progress (step index)")

    # grid & layout
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    # save
    dist_filename = f"MPC_I_meas_distribution_{err}.png"
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, dist_filename), dpi=300)
    plt.close(fig)


    fs = 1.0 / dt
    thd_a = compute_thd(va, fs)
    thd_b = compute_thd(vb, fs)
    thd_c = compute_thd(vc, fs)
    avg_thd = (thd_a + thd_b + thd_c) / 3
    print(f"MPC THD A={thd_a:.2f}%, B={thd_b:.2f}%, C={thd_c:.2f}%, avg={avg_thd:.2f}%")

    # Plot three-phase voltages with THD in title
    plt.figure()
    plt.plot(t, va, label="Va")
    plt.plot(t, vb, label="Vb")
    plt.plot(t, vc, label="Vc")
    # plt.legend()
            
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    # title = (f"Reconstructed Three-Phase Voltages (err: {err}, safety: {safety_const}, sd: {sd_value}, r: {total_disc:.2f}) \n"
    #         f"THD A: {thd_a:.2f}%  B: {thd_b:.2f}%  C: {thd_c:.2f}%  Avg: {avg_thd:.2f}%")
    # plt.title(title)
    three_phase_filename = f"MPC_reconstructed_three_phase_{err}.png"
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plot_dir, three_phase_filename), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== MPC Summary ===")
    print(f"MPC mean return : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"MPC mean violations        : {np.mean(violations):.1f} per episode")

    results_summary.append({
                "alg_type": "FCS-MPC",
                "safety_constraint": False,
                "error_ratio": MPC_error_ratio,
                "seed": "",
                "avg_discounted_return": avg_return,
                "THD_a": thd_a,
                "THD_b": thd_b,
                "THD_c": thd_c,
                "THD_avg": avg_thd
            })


    for dir in seed_dir:
        
        sd_value = dir.split("_sd_")[1].split("_")[0]
        if sd_value not in Best_seeds.get(run, []):
            continue

        alg_type = "FS-RL"
        if "only_RL" in dir and "NRew_True" in dir:
            alg_type = "PPO+MPC(Naive)"
        elif "only_RL" in dir and "Safety_True" in dir:
            alg_type = "PPO+MPC(Modified)"
        elif "only_RL" in dir and "Safety_False" in dir:
            alg_type = "PPO"

        # Find all best_model.zip files
        model_files = [f for f in os.listdir(dir) if f.endswith("best_model.zip")]


        
        sd_value = dir.split("_sd_")[1].split("_")[0]
 

        for model_name in model_files:
            model_path = os.path.join(dir, model_name)
            print(f"\nLoading model: {model_path}")
            model = PPO.load(model_path, env=env)

            # 1. Run multiple episodes to collect I_meas values
            
            discounted_returns = []
            for ep in range(num_episodes):
                obs, info = env.reset()
                terminated = False
                truncated  = False
                total_disc = 0.0
                t_step     = 0
                
                real_vals, imag_vals = [], []
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    i_real, i_imag = obs["I_meas"]
                    real_vals.append(i_real)
                    imag_vals.append(i_imag)
                    total_disc += (gamma ** t_step) * reward
                    t_step += 1

                discounted_returns.append(total_disc)
                print(f"err: {err}, safety: {safety_const}, sd: {sd_value}, ep: {ep+1:3d}, return = {total_disc:.2f}")

            # Plot I_meas distribution
            # ——— build a 0→1 “time” array over the full history ———
            N = len(real_vals)
            time_idx = np.arange(N)

            fig, ax = plt.subplots()
            # draw circle first (so it sits behind)
            circle = plt.Circle((0, 0), radius=30, edgecolor='red', fill=False, linewidth=2)
            ax.add_patch(circle)

            # scatter with clean args
            norm = Normalize(vmin=0, vmax=N-1)
            sc = ax.scatter(
                real_vals, imag_vals,
                marker='o',
                c=time_idx,
                cmap='viridis',
                norm=norm,
                s=10,            # bump size up a bit
                linewidths=0.5,  # thin edges
                alpha=0.8
            )

            # labels, aspect, title
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
            # ax.set_title(
            #     f"The Measured Current Distribution\n"
            #     f"(err: {err}, safety: {safety_const}, sd: {sd_value}, r: {total_disc:.2f})"
            # )

            # colorbar
            cbar = fig.colorbar(sc, ax=ax, label="Progress (step index)")

            # grid & layout
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()

            # save
            dist_filename = f"{alg_type}_I_meas_distribution_{safety_const}_{err}.png"
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(os.path.join(plot_dir, dist_filename), dpi=300)
            plt.close(fig)


            # 2. Run one more episode for three-phase voltages
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            va = info["Voltage_reconstructed_a"][trim_lower:]
            vb = info["Voltage_reconstructed_b"][trim_lower:]
            vc = info["Voltage_reconstructed_c"][trim_lower:]

            fs = 1.0 / dt
            thd_a = compute_thd(va, fs)
            thd_b = compute_thd(vb, fs)
            thd_c = compute_thd(vc, fs)
            avg_thd = (thd_a + thd_b + thd_c) / 3
            print(f"THD A={thd_a:.2f}%, B={thd_b:.2f}%, C={thd_c:.2f}%, avg={avg_thd:.2f}%")

            # Plot three-phase voltages with THD in title
            plt.figure()
            plt.plot(t, va, label="Va")
            plt.plot(t, vb, label="Vb")
            plt.plot(t, vc, label="Vc")
            # plt.legend()
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            # title = (f"Reconstructed Three-Phase Voltages (err: {err}, safety: {safety_const}, sd: {sd_value}, r: {total_disc:.2f}) \n"
            #         f"THD A: {thd_a:.2f}%  B: {thd_b:.2f}%  C: {thd_c:.2f}%  Avg: {avg_thd:.2f}%")
            # plt.title(title)
            three_phase_filename = f"{alg_type}_reconstructed_three_phase_{safety_const}_{err}.png"
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(os.path.join(plot_dir, three_phase_filename), dpi=300, bbox_inches="tight")
            plt.close()

            avg_return = np.mean(discounted_returns)
            results_summary.append({
                "alg_type": alg_type,
                "safety_constraint": safety_const,
                "error_ratio": err,
                "seed": sd_value,
                "avg_discounted_return": avg_return,
                "THD_a": thd_a,
                "THD_b": thd_b,
                "THD_c": thd_c,
                "THD_avg": avg_thd
            })

csv_output_path = os.path.join(plots_base, "summary_results.csv")
csv_fieldnames = [
    "alg_type", "safety_constraint", "error_ratio", "seed",
    "avg_discounted_return", "THD_a", "THD_b", "THD_c", "THD_avg"
]

with open(csv_output_path, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(results_summary)

print(f"\nSaved summary CSV to: {csv_output_path}")



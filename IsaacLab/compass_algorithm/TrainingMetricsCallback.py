import os
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, save_path: str, run_name: str, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.run_name = run_name
        
        # --- Best Model Tracking Variables ---
        self.best_success_rate = -np.inf 
        self.best_runtime = np.inf
        self.best_path_opt = np.inf
        
        # --- Sliding Windows (Evaluates over the last 100 episodes) ---
        self.window_size = 100
        self.success_buffer = deque(maxlen=self.window_size)
        self.crash_buffer = deque(maxlen=self.window_size)
        
        self.opt_buffer = deque(maxlen=self.window_size)
        self.energy_buffer = deque(maxlen=self.window_size)
        self.runtime_buffer = deque(maxlen=self.window_size)
        self.vel_buffer = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        # Loop through the environments (usually just 1 for your setup)
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][idx]
                
                # 1. Reliability Metrics (Always recorded)
                if "is_success" in info:
                    self.success_buffer.append(info["is_success"])
                if "is_crash" in info:
                    self.crash_buffer.append(info["is_crash"])
                    
                # 2. Efficiency Metrics (ONLY recorded if the robot reached the goal)
                if info.get("is_success") == 1.0:
                    if "success_path_optimality" in info:
                        self.opt_buffer.append(info["success_path_optimality"])
                    if "success_energy" in info:
                        self.energy_buffer.append(info["success_energy"])
                    if "success_runtime" in info:
                        self.runtime_buffer.append(info["success_runtime"])
                    if "success_velocity" in info:
                        self.vel_buffer.append(info["success_velocity"])
        return True

    def _on_rollout_end(self) -> None:
        # Only evaluate for the "best" model if we have a statistically significant sample size (e.g., 20 episodes)
        if len(self.success_buffer) >= 20: 
            current_success_rate = np.mean(self.success_buffer)
            
            self.logger.record("rollout_metrics/success_rate", current_success_rate)
            self.logger.record("rollout_metrics/crash_rate", np.mean(self.crash_buffer))
            
            # =================================================================
            # 3-TIER HIERARCHICAL BEST MODEL LOGIC
            # =================================================================
            # Get current efficiencies (default to infinity if buffer is empty)
            curr_runtime = np.mean(self.runtime_buffer) if len(self.runtime_buffer) > 0 else np.inf
            curr_path_opt = np.mean(self.opt_buffer) if len(self.opt_buffer) > 0 else np.inf
            
            save_model = False
            reason = ""

            # TIER 1: Is the success rate strictly better?
            if current_success_rate > self.best_success_rate and current_success_rate > 0.0:
                save_model = True
                reason = f"New Best Success: {current_success_rate:.2%}"
                
            # TIER 2: Success rate is tied. Is the runtime strictly better?
            elif current_success_rate == self.best_success_rate and current_success_rate > 0.0:
                if curr_runtime < self.best_runtime:
                    save_model = True
                    reason = f"Success tied ({current_success_rate:.2%}), FASTER runtime: {curr_runtime:.2f}s"
                    
                # TIER 3: Success AND Runtime are tied. Is the path strictly shorter?
                elif curr_runtime == self.best_runtime:
                    if curr_path_opt < self.best_path_opt:
                        save_model = True
                        reason = f"Success & Runtime tied, SHORTER path: {curr_path_opt:.3f}"

            # If any of the tiers triggered a save...
            if save_model:
                # Update the all-time records
                self.best_success_rate = current_success_rate
                self.best_runtime = curr_runtime
                self.best_path_opt = curr_path_opt
                
                # Save the neural network weights to disk
                best_model_path = os.path.join(self.save_path, f"{self.run_name}_best_model")
                self.model.save(best_model_path)
                
                if self.verbose > 0:
                    print(f"\n[INFO] {reason} | Saving best model...")
            # =================================================================

        # Log efficiency metrics to TensorBoard if we have them
        if len(self.opt_buffer) > 0:
            self.logger.record("rollout_efficiency/path_optimality", np.mean(self.opt_buffer))
            self.logger.record("rollout_efficiency/energy_consumed", np.mean(self.energy_buffer))
            self.logger.record("rollout_efficiency/runtime_seconds", np.mean(self.runtime_buffer))
            self.logger.record("rollout_efficiency/velocity_m_s", np.mean(self.vel_buffer))
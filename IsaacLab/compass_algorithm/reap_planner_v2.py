from __future__ import annotations
from typing import Union
import numpy as np
import sympy as sp


# ==============================================================================
# SECTION 2: THE REAP PLANNER (The Brain)
# ==============================================================================
class REAP_Planner:
    def __init__(self, horizon=15, max_vel=0.25, barrier_beta=100.0, robot_radius=0.3):
        print(">> [REAP] Initializing Symbolic Math (this takes a moment)...")

        # --- PARAMETERS (Now Dynamic) ---
        self.N_HORIZON = horizon
        self.MAX_VEL = max_vel
        self.BARRIER_BETA = barrier_beta
        self.ROBOT_RADIUS = robot_radius
        
        self.DT_SIM = 0.02 
        self.SIM_T_END = 1.0
        self.N_updates = int(self.SIM_T_END / self.DT_SIM)

        # Weights
        # self.Q_mat = np.diag([10.0, 10.0])
        self.Q_mat = np.diag([1.0, 1.0])
        self.R_mat = np.diag([0.5, 0.5])
        # self.P_mat = np.diag([20.0, 20.0])
        self.P_mat = self.Q_mat
        self.target = np.array([0.0, 2.8])

        self.OBSTACLES = np.array([
            [0.0,   0.15, 0.30],
            [-1.3,  0.75, 0.30],
            [0.0,   1.45, 0.30],
            [1.3,   0.75, 0.30],
            [1.3,  -0.45, 0.30],
            [-1.3, -0.45, 0.30],
        ], dtype=np.float32)





        # --- SYMBOLIC MATH SETUP ---
        x_sym = sp.Matrix([sp.Symbol("x"), sp.Symbol("y")])
        u_sym = sp.Matrix([sp.Symbol("u_x"), sp.Symbol("u_y")])

        all_constraints_sym = []
        # Control Constraints
        all_constraints_sym.append(u_sym[0] - self.MAX_VEL)
        all_constraints_sym.append(-u_sym[0] - self.MAX_VEL)
        all_constraints_sym.append(u_sym[1] - self.MAX_VEL)
        all_constraints_sym.append(-u_sym[1] - self.MAX_VEL)

        # Obstacle Constraints
        for i in range(self.OBSTACLES.shape[0]):
            x_i_sym = sp.Matrix(self.OBSTACLES[i, :2])
            r_i = sp.Float(self.OBSTACLES[i, 2])
            r = sp.Float(self.ROBOT_RADIUS)

            diff_vec_sym = x_i_sym - x_sym
            dist_sq_scalar = (diff_vec_sym.T @ diff_vec_sym)[0]
            dist_sym = sp.sqrt(dist_sq_scalar)

            theta_i_sym = 0.5 - (r_i**2 - r**2) / (2 * dist_sq_scalar)
            unit_vec_robot_to_obs_sym = -diff_vec_sym / dist_sym
            b_i_term1_sym = theta_i_sym * x_i_sym + (1 - theta_i_sym) * x_sym
            b_i_term2_sym = r * unit_vec_robot_to_obs_sym
            b_i_scalar = (diff_vec_sym.T @ (b_i_term1_sym + b_i_term2_sym))[0]

            x_next_sym = x_sym + u_sym * sp.Float(self.DT_SIM)
            dot_expr = (diff_vec_sym.T @ x_next_sym)[0, 0]
            all_constraints_sym.append(dot_expr - b_i_scalar)

        self.c_per_step = len(all_constraints_sym)

        # Lambdify (Compile the math to fast functions)
        self.num_constraints_step = sp.lambdify(list(u_sym) + list(x_sym), all_constraints_sym, "numpy")
        dCdu_sym = [c_i.diff(u_var) for c_i in all_constraints_sym for u_var in u_sym]
        self.num_dCdu_step = sp.lambdify(list(u_sym) + list(x_sym), dCdu_sym, "numpy")
        dCdx_sym = [c_i.diff(x_var) for c_i in all_constraints_sym for x_var in x_sym]
        self.num_dCdx_step = sp.lambdify(list(u_sym) + list(x_sym), dCdx_sym, "numpy")

        # --- INTERNAL STATE (Warm Start) ---
        self.opt_vars = np.zeros(self.N_HORIZON * 2)
        self.hat_lambda = np.zeros(self.N_HORIZON * self.c_per_step)

        # Save a "fresh reset" copy (cold-start state)
        self._opt_vars_reset = self.opt_vars.copy()
        self._hat_lambda_reset = self.hat_lambda.copy()

    
    
    def reset(self):
        """Reset REAP warm-start so get_action() depends only on the given state."""
        self.opt_vars[:] = self._opt_vars_reset
        self.hat_lambda[:] = self._hat_lambda_reset



    def get_action(self, current_pos):
        """
        Takes current Robot Position [x, y]
        Returns Optimal Velocity [vx, vy]
        """
        grad_u = self._compute_grad_u(self.opt_vars, current_pos, self.hat_lambda)
        grad_l = self._compute_grad_lambda(self.opt_vars, current_pos)

        # Phi Projection
        phi_val = np.zeros_like(self.hat_lambda)
        for i in range(len(self.hat_lambda)):
            if self.hat_lambda[i] > 1e-10 or (self.hat_lambda[i] <= 1e-10 and grad_l[i] >= 0):
                phi_val[i] = 0
            else:
                phi_val[i] = -grad_l[i]

        checkpoint = grad_l + phi_val
        Sigma = 0.5 # 0.5

        # Update Primal (Control) and Dual (Lambda)
        self.opt_vars = self.opt_vars - Sigma * grad_u
        self.hat_lambda = self.hat_lambda + Sigma * checkpoint
        self.hat_lambda = np.maximum(self.hat_lambda, 0)  # Project >= 0
        self.opt_vars = np.clip(self.opt_vars, -self.MAX_VEL, self.MAX_VEL)

        # Extract first action (Model Predictive Control)
        u_applied = self.opt_vars[:2].copy()

        # Shift Warm Start
        self.opt_vars = np.roll(self.opt_vars, -2)
        self.opt_vars[-2:] = 0.0

        return u_applied

    # --- Internal Helpers ---
    def _get_trajectory(self, u_flat, x0):
        U = u_flat.reshape(self.N_HORIZON, 2)
        X = np.zeros((self.N_HORIZON + 1, 2))
        X[0] = x0
        for k in range(self.N_HORIZON):
            X[k + 1] = X[k] + U[k] * self.DT_SIM
        return U, X

    def _compute_grad_u(self, u_flat, x0, lambdas_flat):
        U, X = self._get_trajectory(u_flat, x0)
        lambdas = lambdas_flat.reshape(self.N_HORIZON, self.c_per_step)
        grad_U = np.zeros((self.N_HORIZON, 2))
        grad_x_accum = np.zeros(2)

        for k in reversed(range(self.N_HORIZON)):
            x_curr, x_next, u_curr = X[k], X[k + 1], U[k]

            grad_control = 2 * self.R_mat @ u_curr
            if k == self.N_HORIZON - 1:
                d_state_cost = 2 * self.P_mat @ (x_next - self.target)
            else:
                d_state_cost = 2 * self.Q_mat @ (x_next - self.target)

            total_grad_x_next = d_state_cost + grad_x_accum

            constraints = np.array(self.num_constraints_step(*u_curr, *x_curr)).flatten()
            dCdu = np.array(self.num_dCdu_step(*u_curr, *x_curr)).reshape(self.c_per_step, 2)
            dCdx = np.array(self.num_dCdx_step(*u_curr, *x_curr)).reshape(self.c_per_step, 2)

            grad_barrier_u = np.zeros(2)
            grad_barrier_x = np.zeros(2)

            for i in range(self.c_per_step):
                log_arg = -constraints[i]
                if log_arg < 1e-6:
                    log_arg = 1e-6
                factor = lambdas[k, i] / log_arg
                grad_barrier_u += factor * dCdu[i]
                grad_barrier_x += factor * dCdx[i]

            grad_U[k] = grad_control + total_grad_x_next * self.DT_SIM + grad_barrier_u
            grad_x_accum = total_grad_x_next + grad_barrier_x

        return grad_U.flatten()

    def _compute_grad_lambda(self, u_flat, x0):
        U, X = self._get_trajectory(u_flat, x0)
        all_grad_lambdas = []
        for k in range(self.N_HORIZON):
            c_val = np.array(self.num_constraints_step(*U[k], *X[k])).flatten()
            log_args = -self.BARRIER_BETA * c_val
            grad_l = -np.log10(np.maximum(log_args, 1e-10))
            grad_l[log_args <= 0.001] = -1e200
            all_grad_lambdas.append(grad_l)
        return np.array(all_grad_lambdas).flatten()
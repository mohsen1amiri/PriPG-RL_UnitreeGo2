import sympy as sp
import numpy as np
import time
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

# ======================================================================
# GLOBAL CONSTANTS & MPC-REAP PARAMETERS
# ======================================================================
# MPC Weights
Q_mat = np.diag([10.0, 10.0])   # State Tracking Weight
R_mat = np.diag([0.5, 0.5])     # Control Effort Weight
P_mat = np.diag([20.0, 20.0])   # Terminal Cost Weight

# REAP Parameters
BARRIER_BETA = 100.0  
SIM_T_END = 80.0     
DT_SIM = 0.01        

# MPC Horizon 
N_HORIZON = 2    # Prediction Horizon (N)

# Target and Initial State
# x_current_init = np.array([-0.01, 3.0]) 
# x_target_2d = np.array([-0.1, -1.0])

############# NEW Target and Initial State #####################
x_current_init = np.array([-0.01, -1.5]) 
x_target_2d = np.array([0,2.5])
# Dynamics
N_STATES = 2       
M_CONTROLS = 2     

# Constraints
MAX_LINEAR_VEL = 2
ROBOT_RADIUS = 0.1
POSITION_TOLERANCE = 0.1

OBSTACLES = np.array([
    ############# NEW Target and Initial State #####################

    [0.0   , .15, .43],  
    [-1.3, .75, 0.43],
    [.0,1.45, 0.43],
    [1.3, .75, 0.43],
    [1.3, -.45, 0.43],
    [-1.3,-.45, 0.43]
############ #####################

    # [0, 1.5, .25],  
    # [.0, 0.0, 0.25],
    # [-.750, 0.750, 0.25],
    # [-.750, -0.750, 0.25],
    # [.750, 0.750, 0.25],
    # [.750, -0.750, 0.25]
])

# ======================================================================
# SYMBOLIC SETUP (Single Step Primitives)
# ======================================================================

# --- Symbolic Variables ---
x_sym = sp.Matrix([sp.Symbol('x'), sp.Symbol('y')])
u_sym = sp.Matrix([sp.Symbol('u_x'), sp.Symbol('u_y')])

# --- Constraint Set C(u, x) <= 0 ---
all_constraints_sym = []

# 1. Control Constraints 
all_constraints_sym.append(u_sym[0] - MAX_LINEAR_VEL)
all_constraints_sym.append(-u_sym[0] - MAX_LINEAR_VEL)
all_constraints_sym.append(u_sym[1] - MAX_LINEAR_VEL)
all_constraints_sym.append(-u_sym[1] - MAX_LINEAR_VEL)

# 2. Obstacle Constraints 
for i in range(OBSTACLES.shape[0]):
    x_i_sym = sp.Matrix(OBSTACLES[i, :2])
    r_i = sp.Float(OBSTACLES[i, 2])
    r = sp.Float(ROBOT_RADIUS)

    diff_vec_sym = x_i_sym - x_sym
    dist_sq_scalar = (diff_vec_sym.T @ diff_vec_sym)[0]
    dist_sym = sp.sqrt(dist_sq_scalar)
    
    # Hyperplane Barrier parameters
    theta_i_sym = 0.5 - (r_i**2 - r**2) / (2 * dist_sq_scalar)
    unit_vec_robot_to_obs_sym = -diff_vec_sym / dist_sym

    b_i_term1_sym = theta_i_sym * x_i_sym + (1 - theta_i_sym) * x_sym
    b_i_term2_sym = r * unit_vec_robot_to_obs_sym
    b_i_sym = diff_vec_sym.T @ (b_i_term1_sym + b_i_term2_sym)
    b_i_scalar = b_i_sym[0]

    x_next_sym = x_sym + u_sym * sp.Float(DT_SIM) # Important: Constraints checked at x_{k+1}
    dot_expr = (diff_vec_sym.T @ x_next_sym)[0, 0]
    constraint_value_sym = dot_expr - b_i_scalar

    all_constraints_sym.append(constraint_value_sym)

c_per_step = len(all_constraints_sym)

# --- Lambdify Gradients ---
# We need derivatives w.r.t U (direct effect) AND X (chain rule effect)
num_constraints_step = sp.lambdify(list(u_sym) + list(x_sym), all_constraints_sym, 'numpy')

dCdu_sym = [c_i.diff(u_var) for c_i in all_constraints_sym for u_var in u_sym]
num_dCdu_step = sp.lambdify(list(u_sym) + list(x_sym), dCdu_sym, 'numpy')

dCdx_sym = [c_i.diff(x_var) for c_i in all_constraints_sym for x_var in x_sym]
num_dCdx_step = sp.lambdify(list(u_sym) + list(x_sym), dCdx_sym, 'numpy')

# ======================================================================
# REAP GRADIENT FUNCTIONS (Full Horizon Chain Rule)
# ======================================================================

def get_trajectory(u_flat, x0):
    """Forward simulate dynamics to get state trajectory."""
    U = u_flat.reshape(N_HORIZON, M_CONTROLS)
    X = np.zeros((N_HORIZON + 1, N_STATES))
    X[0] = x0
    for k in range(N_HORIZON):
        X[k+1] = X[k] + U[k] * DT_SIM
    return U, X

def num_grad_u(u_flat, x0, lambdas_flat):
    """
    Calculates gradient of Lagrangian w.r.t control sequence U.
    Includes Cost Gradient + Barrier Gradient using Chain Rule.
    """
    U, X = get_trajectory(u_flat, x0)
    lambdas = lambdas_flat.reshape(N_HORIZON, c_per_step)
    
    grad_U = np.zeros((N_HORIZON, M_CONTROLS))
    
    # We accumulate the "cost-to-go" gradient (adjoint/backprop)
    # dJ/dx at step k
    grad_x_accumulated = np.zeros(N_STATES) 

    # Iterate BACKWARDS from N-1 down to 0
    for k in reversed(range(N_HORIZON)):
        x_curr = X[k]   # x_k
        x_next = X[k+1] # x_{k+1}
        u_curr = U[k]
        
        # --- 1. Cost Gradient Terms ---
        # A. Direct Control Cost: d(uRs)/du = 2Ru
        grad_control_cost = 2 * R_mat @ u_curr
        
        # B. State Cost Gradient (Tracking)
        # If k == N-1, we hit the Terminal Cost P
        if k == N_HORIZON - 1:
            d_state_cost_dx = 2 * P_mat @ (x_next - x_target_2d)
        else:
            # Intermediate Q cost
            d_state_cost_dx = 2 * Q_mat @ (x_next - x_target_2d)
            
        # Add accumulated gradient from future steps
        total_grad_x_next = d_state_cost_dx + grad_x_accumulated
        
        # --- 2. Barrier Gradient Terms ---
        # Constraints are C(u_k, x_k) <= 0 (approximated at x_{k+1} usually)
        # Evaluate constraints at this step
        constraints_val = np.array(num_constraints_step(*u_curr, *x_curr)).flatten()
        
        # Jacobians for this step
        dCdu_val = np.array(num_dCdu_step(*u_curr, *x_curr)).reshape(c_per_step, M_CONTROLS)
        dCdx_val = np.array(num_dCdx_step(*u_curr, *x_curr)).reshape(c_per_step, N_STATES)
        
        grad_barrier_u = np.zeros(M_CONTROLS)
        grad_barrier_x = np.zeros(N_STATES)
        
        for i in range(c_per_step):
            log_arg = -constraints_val[i]
            if log_arg < 1e-6: log_arg = 1e-6 # Numerical safety
            
            factor = lambdas[k, i] / log_arg
            
            grad_barrier_u += factor * dCdu_val[i]
            grad_barrier_x += factor * dCdx_val[i]
            
        # --- 3. Combine to get dL/du_k ---
        # Chain rule: x_{k+1} = x_k + u_k * dt
        # dx_{k+1}/du_k = dt * I
        
        # Gradient of Objective w.r.t u_k = dCost/du + (dCost/dx_{next} * dx_{next}/du)
        grad_U[k] = grad_control_cost + total_grad_x_next * DT_SIM + grad_barrier_u
        
        # --- 4. Update Accumulator for next backward step (dL/dx_k) ---
        # dx_{k+1}/dx_k = I (for simple linear model x' = x + u)
        # Gradient flows back: grad_x_k = grad_barrier_x + grad_x_{k+1} * (dx_{k+1}/dx_k)
        grad_x_accumulated = total_grad_x_next + grad_barrier_x

    return grad_U.flatten()

def num_grad_lambda(u_flat, x0):
    """Gradient w.r.t Lagrange multipliers for ALL steps."""
    U, X = get_trajectory(u_flat, x0)
    all_grad_lambdas = []
    
    for k in range(N_HORIZON):
        u_curr = U[k]
        x_curr = X[k]
        
        # Evaluate constraints
        constraints_val = np.array(num_constraints_step(*u_curr, *x_curr)).flatten()
        
        log_args = -BARRIER_BETA * constraints_val
        grad_l = -np.log10(np.maximum(log_args, 1e-10))
        
        # If constraint is well satisfied, gradient is small
        # If constraint is violated (log_args < 0), this formulation handles it via maximum
        grad_l[log_args <= 0.001] = -1e200 # Hard push
        
        all_grad_lambdas.append(grad_l)
        
    return np.array(all_grad_lambdas).flatten()

def Phi(hatLambda, grad_B_lambda):
    phi = np.zeros_like(hatLambda, dtype=float)
    for i in range(len(hatLambda)):
        if hatLambda[i] > 1e-10 or (hatLambda[i] <= 1e-10 and grad_B_lambda[i] >= 0):
            phi[i] = 0
        else:
            phi[i] = -grad_B_lambda[i]
    return phi


# ======================================================================
# MAIN LOOP
# ======================================================================
def run_simulation():
    print("--- 🏃‍♂️ Starting Full-Horizon MPC-REAP ---")
    
    x_current = x_current_init.copy()
    # Optimization variables: entire sequence U [u0, u1, ... uN-1]
    opt_vars = np.zeros(N_HORIZON * M_CONTROLS) 
    # Lambda variables: for every constraint at every step
    total_constraints = N_HORIZON * c_per_step
    hat_lambda = np.zeros(total_constraints)
    
    x_hist = [x_current.copy()]
    u_hist = []
    t_hist = [0.0]
    
    steps = int(SIM_T_END / DT_SIM)
    
    for k in range(steps):
        if np.linalg.norm(x_current - x_target_2d) < POSITION_TOLERANCE:
            print("Goal Reached.")
            break
            
        # 1. Initialization
        if k == 0:
            opt_vars=np.zeros((2 * N_HORIZON,1)).flatten()
        else:
            # REAP Step
            # Calculate gradients based on FULL horizon
            grad_u_val = num_grad_u(opt_vars, x_current, hat_lambda)
            grad_l_val = num_grad_lambda(opt_vars, x_current)
            
            # Checkpoint calculation
            phi_val = Phi(hat_lambda, grad_l_val)
            checkpoint = grad_l_val + phi_val
            
            # Update Step with simple line search logic
            Sigma = 0.5 # Learning rate
            
            # Primal-Dual Update
            opt_vars = opt_vars - Sigma * grad_u_val
            hat_lambda = hat_lambda + Sigma * checkpoint
            
            # Project Lambda >= 0
            hat_lambda = np.maximum(hat_lambda, 0)
            
            # Clip controls to bounds (Simple projection)
            opt_vars = np.clip(opt_vars, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)

        # 2. Apply Control
        # Extract first control u_0 from the sequence
        u_applied = opt_vars[:M_CONTROLS]
        
        # 3. Step Dynamics
        x_current = x_current + u_applied * DT_SIM
        
        # 4. Shift Warm Start
        # Move u1 -> u0, u2 -> u1, ... uN -> uN-1, and append 0 at end
        opt_vars = np.roll(opt_vars, -M_CONTROLS)
        opt_vars[-M_CONTROLS:] = 0.0
        
        x_hist.append(x_current.copy())
        u_hist.append(u_applied.copy())
        t_hist.append((k+1)*DT_SIM)
        
    return np.array(x_hist), np.array(u_hist), np.array(t_hist)

if __name__ == "__main__":
    x_h, u_h, t_h = run_simulation()
        
    # Quick Plot
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(x_h[:,0], x_h[:,1], label='Path')
    plt.scatter(x_target_2d[0], x_target_2d[1], c='r', label='Target')
    for obs in OBSTACLES:
        c = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.3)
        plt.gca().add_patch(c)
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(t_h[:-1], u_h)
    plt.title('Controls')
    plt.grid(True)
    plt.show()
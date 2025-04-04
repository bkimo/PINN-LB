# Previous Version: lb_inf_LHS_20250402
# Differrences:                  
#    Averaged L2 norm error --> Averaged Relative L2 norm error over time
#
#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import Colorbar
import os
from multiprocessing import cpu_count
from time import time
import datetime

# =============================================================================
# Plot settings
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rc('font', family='serif', size=14)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=14)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

# =============================================================================
# GPU Configuration
print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")
print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Logical CPU cores: {cpu_count()}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        selected_gpus = gpus[:1]  
        tf.config.set_visible_devices(selected_gpus, 'GPU')
        for gpu in selected_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPUs: " + ", ".join([gpu.name for gpu in selected_gpus]))
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")
else:
    print("No GPUs available. Using CPU")

# =============================================================================
# Set data type and reproducibility
DTYPE = 'float32'
DTYPE_SV = 'f32'
#DTYPE = 'float64' #Caution! This will consume lots of com memories.
#DTYPE_SV = 'f64'
tf.keras.backend.set_floatx(DTYPE)
np.random.seed(1234)
tf.random.set_seed(1234)

# =============================================================================
# Constants
pi = tf.constant(np.pi, dtype=DTYPE)

# ====================================================================== =======
# Set IB and domain bounds
IB = 1

# Option: use fixed or trainable alpha2.
FIX_ALPHA2 = False   # Set True for a fixed (non-trainable) alpha2 
                    # False means alpha2 = alpha2(t) trainable over time (ephocs)
ALPHA = 0.03
ALPHA2_VALUE = ALPHA**2

# =============================================================================
# Training Configuration
TRAINING_CONFIG = {
    'N_epoch': 20,  # Adjust as needed
    'N_0': 1000,
    'N_b': 1000, 
    'N_r': 10000,
    'num_hidden_layers': 8,
    'num_neurons_per_layer': 20,
    'initial_learning_rate': 1e-2,
    # Decay steps at 1/3 and 2/3 of total epochs
    'decay_steps': [lambda epochs: epochs // 3,
                   lambda epochs: epochs // 3 + (2 * (epochs - epochs // 3)) // 3],
    'decay_rates': [1e-2, 1e-4, 1e-5],  # Initial rate, rate after first decay, rate after second decay
    'USE_DYNAMIC_WEIGHTS': False,  # Option to toggle dynamic weighting
    'initial_weights': {'residual': 1.0, 'initial': 1.0, 'boundary': 1.0},  # Fixed weights when dynamic weights are disabled
    'weight_update_frequency': 1000,  # How often to update weights when dynamic weighting is enabled
    'weight_smoothing_factor': 0.7  # Smoothing factor for dynamic weights
}

# Calculate actual decay steps based on total epochs
decay_steps = [step_fn(TRAINING_CONFIG['N_epoch']) for step_fn in TRAINING_CONFIG['decay_steps']]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    decay_steps, TRAINING_CONFIG['decay_rates']
)
model_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# =============================================================================
# Data and results directories
DATA_DIR = './data'
RESULT_DIR = './output/fig13'
# Create results directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)

# Default domain bounds
DEFAULT_TMIN = 0.0
DEFAULT_TMAX = 2.0
DEFAULT_XMIN = -3.0
DEFAULT_XMAX = 3.0
# Default test grid dimensions
DEFAULT_NT = 400
DEFAULT_NX = 1000

def load_exact_data():
    """
    Load exact solution data from files.
    
    Attempts to load pre-computed exact solution data for the specified IB value.
    If data is not found, returns default domain bounds and grid dimensions.
    
    Returns:
        tuple: (exact_t, exact_x, exact_v, tmin, tmax, xmin, xmax, Nt, Nx)
            exact_t (ndarray): Time coordinates of exact solution
            exact_x (ndarray): Spatial coordinates of exact solution
            exact_v (ndarray): Values of exact solution
            tmin (float): Minimum time value
            tmax (float): Maximum time value
            xmin (float): Minimum spatial value
            xmax (float): Maximum spatial value
            Nt (int): Number of time points
            Nx (int): Number of spatial points
    """
    try:
   #     exact_t = np.load(os.path.join(DATA_DIR, f'exact_data{IB}_{DTYPE_SV}_t.npy'))
   #     exact_x = np.load(os.path.join(DATA_DIR, f'exact_data{IB}_{DTYPE_SV}_x.npy'))
   #     exact_v = np.load(os.path.join(DATA_DIR, f'exact_data{IB}_{DTYPE_SV}_v.npy'))

        exact_t = np.load(os.path.join(DATA_DIR, f'exact_longdata{IB}_{DTYPE_SV}_t.npy'))
        exact_x = np.load(os.path.join(DATA_DIR, f'exact_longdata{IB}_{DTYPE_SV}_x.npy'))
        exact_v = np.load(os.path.join(DATA_DIR, f'exact_longdata{IB}_{DTYPE_SV}_v.npy'))
        
        # Get domain bounds from data
        tmin, tmax = float(np.min(exact_t)), float(np.max(exact_t))
        xmin, xmax = float(np.min(exact_x)), float(np.max(exact_x))
        
        # Get grid dimensions from data
        Nt = len(np.unique(exact_t))
        Nx = len(np.unique(exact_x))
        
        print(f"Successfully loaded exact data for IB={IB}")
        print(f"Domain bounds from data: t ∈ [{tmin}, {tmax}], x ∈ [{xmin}, {xmax}]")
        print(f"Grid dimensions from data: Nt={Nt}, Nx={Nx}")
        return exact_t, exact_x, exact_v, tmin, tmax, xmin, xmax, Nt, Nx
    except FileNotFoundError as e:
        print(f"Warning: Exact solution data files not found for IB={IB}. Using analytical solution if available.")
        print(f"Looked in directory: {os.path.abspath(DATA_DIR)}")
        print(f"Expected files: exact_data{IB}_t.npy, exact_data{IB}_x.npy, exact_data{IB}_v.npy")
        print(f"Using default domain bounds: t ∈ [{DEFAULT_TMIN}, {DEFAULT_TMAX}], x ∈ [{DEFAULT_XMIN}, {DEFAULT_XMAX}]")
        print(f"Using default grid dimensions: Nt={DEFAULT_NT}, Nx={DEFAULT_NX}")
        return None, None, None, DEFAULT_TMIN, DEFAULT_TMAX, DEFAULT_XMIN, DEFAULT_XMAX, DEFAULT_NT, DEFAULT_NX

# Get exact data, domain bounds, and grid dimensions
exact_t, exact_x, exact_v, tmin, tmax, xmin, xmax, Nt_test_grid, Nx_test_grid = load_exact_data()
lb = tf.constant([tmin, xmin], dtype=DTYPE)
ub = tf.constant([tmax, xmax], dtype=DTYPE)

print(f"Using test grid dimensions: Nt={Nt_test_grid}, Nx={Nx_test_grid}")

# =============================================================================
# Initial and Boundary Conditions
def get_initial_boundary_conditions(IB, xmin, xmax, DTYPE, pi):
    """
    Get initial and boundary condition functions for the specified IB value.
    
    Parameters:
        IB (int): Initial/boundary condition index (0-7)
        xmin (float): Minimum spatial domain value
        xmax (float): Maximum spatial domain value
        DTYPE: Data type for TensorFlow operations
        pi (tf.Tensor): Pi constant with appropriate data type
    
    Returns:
        tuple: (initial_condition, boundary_condition)
            initial_condition (function): Function that takes x and returns initial values
            boundary_condition (function): Function that takes t, x and returns boundary values
    """
    if IB == 0:
        def initial_condition(x):
            return -tf.sin(pi * x)
        def boundary_condition(t, x):
            return tf.zeros((tf.shape(x)[0], 1), dtype=DTYPE)
    elif IB == 1:
        def initial_condition(x):
            return tf.where(x <= 0, tf.constant(1.0, dtype=DTYPE), tf.constant(0.0, dtype=DTYPE))
        def boundary_condition(t, x):
            return tf.where(tf.equal(x, tf.constant(xmin, dtype=DTYPE)),
                            tf.constant(1.0, dtype=DTYPE),
                            tf.constant(0.0, dtype=DTYPE))
    elif IB == 2:
        def initial_condition(x):
            return tf.where(x <= 0, tf.constant(1.0, dtype=DTYPE),
                            tf.where(x < 1.0, tf.constant(1.0, dtype=DTYPE) - x,
                                     tf.constant(0.0, dtype=DTYPE)))
        def boundary_condition(t, x):
            return tf.where(tf.equal(x, tf.constant(xmin, dtype=DTYPE)),
                            tf.constant(1.0, dtype=DTYPE),
                            tf.constant(0.0, dtype=DTYPE))
    elif IB == 3:
        def initial_condition(x):
            return tf.where(x <= 0, tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE))
        def boundary_condition(t, x):
            return tf.where(tf.equal(x, tf.constant(xmin, dtype=DTYPE)),
                            tf.constant(0.0, dtype=DTYPE),
                            tf.constant(1.0, dtype=DTYPE))
    elif IB == 4:
        def initial_condition(x):
            return tf.where(x < 0, tf.constant(0.0, dtype=DTYPE),
                            tf.where(x <= 1.0, tf.constant(1.0, dtype=DTYPE),
                                     tf.constant(0.0, dtype=DTYPE)))
        def boundary_condition(t, x):
            return tf.where(tf.equal(x, tf.constant(xmin, dtype=DTYPE)),
                            tf.constant(0.0, dtype=DTYPE),
                            tf.constant(1.0, dtype=DTYPE))
    elif IB == 5:
        def initial_condition(x):
            return tf.where(x < 0, tf.constant(2.0, dtype=DTYPE),
                            tf.where(x < 1.0, tf.constant(1.5, dtype=DTYPE),
                                     tf.where(x < 2.0, tf.constant(1.0, dtype=DTYPE),
                                              tf.where(x < 3.0, tf.constant(0.5, dtype=DTYPE),
                                                       tf.constant(0.0, dtype=DTYPE)))))
        def boundary_condition(t, x):
            return tf.where(tf.equal(x, tf.constant(xmin, dtype=DTYPE)),
                            tf.constant(2.0, dtype=DTYPE),
                            tf.constant(0.0, dtype=DTYPE))
    elif IB == 6:
        def initial_condition(x):
            return tf.where(x <= -1.0, tf.constant(4.0, dtype=DTYPE),
                            tf.where(x <= 1.0, tf.constant(-1.5, dtype=DTYPE),
                                     tf.constant(0.5, dtype=DTYPE)))
        def boundary_condition(t, x):
            # Only enforce inflow boundary condition at xmin
            inflow_mask = tf.equal(x, tf.constant(xmin, dtype=DTYPE))
            inflow_value = tf.constant(4.0, dtype=DTYPE)
            # For outflow (x = xmax), use zero weight in loss function
            outflow_mask = tf.equal(x, tf.constant(xmax, dtype=DTYPE))
            # Return a tuple of (values, weights)
            values = tf.where(inflow_mask, inflow_value, tf.zeros_like(x, dtype=DTYPE))
            weights = tf.cast(inflow_mask, dtype=DTYPE)
            return values, weights
    elif IB == 7:
        def initial_condition(x):
            return tf.where(x <= -1.0, tf.constant(4.0, dtype=DTYPE),
                            tf.where(x <= 1.0, tf.constant(2.0, dtype=DTYPE),
                                     tf.constant(0.0, dtype=DTYPE)))
        def boundary_condition(t, x):
            # Only enforce inflow boundary condition at xmin
            inflow_mask = tf.equal(x, tf.constant(xmin, dtype=DTYPE))
            inflow_value = tf.constant(4.0, dtype=DTYPE)
            # For outflow (x = xmax), use zero weight in loss function
            outflow_mask = tf.equal(x, tf.constant(xmax, dtype=DTYPE))
            # Return a tuple of (values, weights)
            values = tf.where(inflow_mask, inflow_value, tf.zeros_like(x, dtype=DTYPE))
            weights = tf.cast(inflow_mask, dtype=DTYPE)
            return values, weights
    else:
        print(f"Warning: Invalid IB value {IB}, using default zero conditions")
        def initial_condition(x):
            return tf.zeros_like(x, dtype=DTYPE)
        def boundary_condition(t, x):
            return tf.zeros_like(t, dtype=DTYPE)
    return initial_condition, boundary_condition

# =============================================================================
# True Solution Function
def get_true_solution(IB):
    """
    Get the analytical true solution function for the specified IB value.
    
    Parameters:
        IB (int): Initial/boundary condition index (0-7)
    
    Returns:
        function: Function that takes X (t,x coordinates) and returns solution values
    """
    if IB == 0:
        def u_true(X):
            t, x = X[:, 0:1], X[:, 1:2]
            return -tf.sin(tf.constant(np.pi, dtype=DTYPE) * x)
            
    elif IB == 1:
        def u_true(X):
            t, x = X[:, 0:1], X[:, 1:2]
            cond1 = tf.cast(tf.math.less_equal(x, 0.5 * t), dtype=DTYPE)
            cond2 = tf.cast(tf.math.greater(x, 0.5 * t), dtype=DTYPE)
            a = tf.math.multiply(cond1, tf.constant(1.0, dtype=DTYPE))
            b = tf.math.multiply(cond2, tf.constant(0.0, dtype=DTYPE))
            return a + b
    elif IB == 2:
        def u_true(X):
            t, x = X[:, 0:1], X[:, 1:2]
            t_safe = tf.where(t == tf.constant(1., dtype=DTYPE), 
                              tf.constant(1.0, dtype=DTYPE) 
                                  - tf.constant(1e-10, dtype=DTYPE), t)
            denom = 1 - t_safe
            b_part = tf.where((x < tf.constant(1.0, dtype=DTYPE)) & (x > t), 
                              (tf.constant(1.0, dtype=DTYPE) - x)/denom, 
                               tf.constant(0., dtype=DTYPE))
            a_part = tf.where(x <= t, tf.constant(1.0, dtype=DTYPE), 
                               tf.constant(0., dtype=DTYPE))
            c_part = tf.where(x >= tf.constant(1.0, dtype=DTYPE), 
                              tf.constant(0., dtype=DTYPE), 
                              tf.constant(0., dtype=DTYPE))
            return a_part + b_part + c_part
        return u_true   
    elif IB == 3:
        def u_true(X):
            t, x = X[:, 0:1], X[:, 1:2]
            t_safe = t + tf.constant(1e-10, dtype=DTYPE)
#            t_safe = tf.where(t == tf.constant(1., dtype=DTYPE), 
#                              tf.constant(1.0, dtype=DTYPE)-tf.constant(1e-10, dtype=DTYPE), t)
            cond1 = tf.cast(x <= tf.constant(0.0, dtype=DTYPE), DTYPE)
            cond2 = tf.cast((x < t_safe) & (x > tf.constant(0.0, dtype=DTYPE)), DTYPE)
            cond3 = tf.cast(x >= t_safe, DTYPE)
            a = tf.multiply(cond1, tf.constant(0.0, dtype=DTYPE))
            b = tf.multiply(cond2, x / t_safe)
            c = tf.multiply(cond3, tf.constant(1.0, dtype=DTYPE))
            return a + b + c
        return u_true
    elif IB == 4:  
        def u_true(X):
            t, x = X[:, 0:1], X[:, 1:2]
            t_safe = tf.where(t == 0, 1e-10, t)
            cond00 = tf.cast((x < 0.0) & (t == 0.0), DTYPE)
            cond01 = tf.cast((x <= 1) & (x >= 0.0) & (t == 0.0), DTYPE)
            cond02 = tf.cast((x > 1.0) & (t == 0.0), DTYPE)
            cond10 = tf.cast((x < 0.0) & (t >= 0.0) & (t < 2.0), DTYPE)
            cond11 = tf.cast((x < t) & (x >= 0.0) & (t > 0.0) & (t < 2.0), DTYPE)
            cond12 = tf.cast((x >= t) & (x <= 1.0 + t/2.0) & (t > 0.0) & (t < 2.0), DTYPE)
            cond13 = tf.cast((x > 1.0 + t/2.0) & (t > 0.0) & (t < 2.0), DTYPE)
            cond20 = tf.cast((x <= 0.0) & (t >= 2.0), DTYPE)
            cond21 = tf.cast((x >= 0.0) & (x <= tf.sqrt(2.0*t)) & (t >= 2.0), DTYPE)
            cond22 = tf.cast((x > tf.sqrt(2.0*t)) & (t >= 2.0), DTYPE)
            a = tf.multiply(cond00, 0.0)
            b = tf.multiply(cond01, 1.0)
            c = tf.multiply(cond02, 0.0)
            d = tf.multiply(cond10, 0.0)
            e = tf.multiply(cond11, x / t_safe)
            f = tf.multiply(cond12, 1.0)
            g = tf.multiply(cond13, 0.0)
            h = tf.multiply(cond20, 0.0)
            i = tf.multiply(cond21, x / t_safe)
            j = tf.multiply(cond22, 0.0)
            result = tf.where(
                t == 0.0,
                a + b + c,
                tf.where(
                    (t > 0.0) & (t < 2.0),
                    d + e + f + g,
                    h + i + j
                )
            )
            return result
        return u_true
    elif IB == 66: # My solution
        def u_true(X):
            t, x = X[:, 0:1], X[:, 1:2]
            """Exact solution for IB=6 with shock and rarefaction, t protection."""
            # Critical time when shock meets rarefaction
            t_c = 8.0 / 11.0
            # Shock position
            x_s = tf.where(t < t_c, -1.0 + 1.25 * t,
                            4.0 * t + 1.0 - tf.sqrt(22.0) * tf.sqrt(t)
                           )
            u_initial = tf.where(x < -1.0, tf.constant(4.0, dtype=DTYPE),
                                 tf.where(x < 1.0, tf.constant(-1.5, dtype=DTYPE), 
                                                          tf.constant(0.5, dtype=DTYPE)
                                         )
                                )

            t_safe = tf.where(t == 0, 1e-10, t)
            u = tf.zeros_like(x, dtype=DTYPE)

            # Rarefaction solution
            u_rarefaction = (x - 1.0) / (t + 1e-9)  # Small epsilon to avoid division by zero
            # Full solution
            u = tf.where(t == 0.0, u_initial,
                        tf.where(x < x_s, tf.constant(4.0, dtype=DTYPE),
                                tf.where(x < 1.0 - 1.5 * t, tf.constant(-1.5, dtype=DTYPE),
                                    # Only applies when t < t_c and x_s < 1.0 - 1.5 t
                                        tf.where(x <= 1.0 + 0.5 * t, u_rarefaction, 
                                                 tf.constant(0.5, dtype=DTYPE)
                                                            )
                                                )
                                )
                        )
            return u
        return u_true
    elif IB == 6:  # Dr. Lee's solution
        def u_true(X):
            t, x = X[:, 0:1], X[:, 1:2]
            """Exact solution for IB=6 with shock and rarefaction, t protection."""
            # Ensure inputs are float32 tensors
            x = tf.cast(x, dtype=tf.float32)
            t = tf.cast(t, dtype=tf.float32)
            
            # Shock position: x_s(t) = -1 + 1.25t
            shock_position = -1.0 + 1.25 * t
            
            # Rarefaction boundaries
            rarefaction_left = 1.0 - 1.5 * t   # x = 1 - 1.5t
            rarefaction_right = 1.0 + 0.5 * t  # x = 1 + 0.5t
            
            # Shock-rarefaction interaction time
            t_interaction = 2.0 / 2.75  # ~0.727
            
            # Define the solution piecewise
            def solution_pre_interaction():
                # Before shock reaches rarefaction (t < 0.727)
                conditions = [
                    x < shock_position,                          # Left of shock
                    tf.logical_and(x >= shock_position, 
                                x < rarefaction_left),          # Between shock and rarefaction left
                    tf.logical_and(x >= rarefaction_left, 
                                x < rarefaction_right),         # Rarefaction region
                    x >= rarefaction_right                       # Right of rarefaction
                ]
                values = [
                    tf.constant(4.0, dtype=tf.float32),          # v = 4.0
                    tf.constant(-1.5, dtype=tf.float32),         # v = -1.5
                    (x - 1.0) / t,                               # v = (x - 1)/t (rarefaction)
                    tf.constant(0.5, dtype=tf.float32)           # v = 0.5
                ]
                return tf.where(conditions[0], values[0],
                                tf.where(conditions[1], values[1],
                                        tf.where(conditions[2], values[2],
                                                values[3])))

            def solution_post_interaction():
                # After shock enters rarefaction (simplified, assumes shock continues)
                # Shock speed may need adjustment; here we extend with v = 0.5 beyond
                conditions = [
                    x < shock_position,                          # Left of shock
                    tf.logical_and(x >= shock_position, 
                                x < rarefaction_right),         # Between shock and rarefaction right
                    x >= rarefaction_right                       # Right of rarefaction
                ]
                values = [
                    tf.constant(4.0, dtype=tf.float32),          # v = 4.0
                    tf.where(x < rarefaction_left, 
                            tf.constant(-1.5, dtype=tf.float32),
                            (x - 1.0) / t),                      # Transition through rarefaction
                    tf.constant(0.5, dtype=tf.float32)           # v = 0.5
                ]
                return tf.where(conditions[0], values[0],
                                tf.where(conditions[1], values[1],
                                        values[2]))

            # Handle t = 0 separately to avoid division by zero
            v_initial = tf.where(x <= -1.0, tf.constant(4.0, dtype=tf.float32),
                                tf.where(x <= 1.0, tf.constant(-1.5, dtype=tf.float32),
                                        tf.constant(0.5, dtype=tf.float32)))
            
            # Select solution based on time
            u = tf.where(tf.equal(t, 0.0), v_initial,
                        tf.where(t < t_interaction, 
                                solution_pre_interaction(),
                                solution_post_interaction()))
            
            return u
        return u_true
    elif IB == 7:
        def u_true(X):
            """
            Exact solution for the 1D inviscid Burgers equation with given initial and boundary conditions.
            Handles two shocks merging at t = 1.
            """
            t, x = X[:, 0:1], X[:, 1:2]  # Extract t and x from input tensor X
    
            # Critical time when the two shocks meet
            t_c = tf.constant(1.0, dtype=DTYPE)
    
            # Shock positions
            # For t < 1: two shocks at x = -1 + 3t and x = 1 + t
            # For t >= 1: single shock at x = 2t
            x_s1 = -1.0 + 3.0 * t  # First shock position (speed 3)
            x_s2 = 1.0 + t         # Second shock position (speed 1)
            x_s_merged = 2.0 * t   # Merged shock position (speed 2)
    
            # Initial condition at t = 0
            u_initial = tf.where(x <= -1.0, tf.constant(4.0, dtype=DTYPE),
                                 tf.where(x <= 1.0, tf.constant(2.0, dtype=DTYPE),
                                          tf.constant(0.0, dtype=DTYPE)))
    
            # Protect against t = 0 for division or comparisons
            t_safe = tf.where(t == 0, tf.constant(1e-10, dtype=DTYPE), t)
    
            # Initialize output tensor
            u = tf.zeros_like(x, dtype=DTYPE)
    
            # Full solution
            u = tf.where(t == 0.0, u_initial,  # Apply initial condition at t = 0
                         tf.where(t < t_c,
                                  # For t < 1: two shocks
                                  tf.where(x < x_s1, tf.constant(4.0, dtype=DTYPE),
                                           tf.where(x < x_s2, tf.constant(2.0, dtype=DTYPE),
                                                    tf.constant(0.0, dtype=DTYPE))),
                                # For t >= 1: single shock
                                  tf.where(x < x_s_merged, tf.constant(4.0, dtype=DTYPE),
                                          tf.constant(0.0, dtype=DTYPE))))
            return u
        return u_true
    else:
        def u_true(X):
            print(f"Warning: No analytical solution available for IB={IB}. Using zeros.")
            return tf.zeros((tf.shape(X)[0], 1), dtype=DTYPE)
        return u_true              
    return u_true

# =============================================================================
# Alpha2 Network
class Alpha2Net(tf.keras.Model):
    """
    Neural network for modeling the diffusion-like parameter alpha2.
    
    Can be configured as either a fixed constant or a trainable function of time.
    
    Attributes:
        trainable_alpha2 (bool): Whether alpha2 is trainable
        alpha2 (tf.Variable): Fixed alpha2 value (used when trainable_alpha2=False)
        alpha2_min (tf.Tensor): Minimum allowed alpha2 value
        alpha2_max (tf.Tensor): Maximum allowed alpha2 value
        hidden_layers (list): List of dense layers (used when trainable_alpha2=True)
        output_layer (tf.keras.layers.Dense): Output layer (used when trainable_alpha2=True)
    """
    
    def __init__(self, trainable_alpha2=True, initial_value=0.009, hidden_layers=3, neurons=20):
        """
        Initialize the alpha2 network.
        
        Parameters:
            trainable_alpha2 (bool): Whether alpha2 is trainable
            initial_value (float): Initial value for alpha2
            hidden_layers (int): Number of hidden layers (for trainable alpha2)
            neurons (int): Number of neurons per hidden layer (for trainable alpha2)
        """
        super(Alpha2Net, self).__init__()
        # Define bounds as constants
        self.alpha2_min = tf.constant(1e-5, dtype=DTYPE)
        self.alpha2_max = tf.constant(1e-1, dtype=DTYPE)
        self.trainable_alpha2 = trainable_alpha2
        
        # Always create alpha2 attribute for compatibility
        self.alpha2 = tf.Variable(initial_value, dtype=DTYPE, trainable=False)
        
        if trainable_alpha2:
            # Neural network for time-dependent alpha2
            self.hidden_layers = []
            for _ in range(hidden_layers):
                self.hidden_layers.append(tf.keras.layers.Dense(neurons, activation='tanh'))
            self.output_layer = tf.keras.layers.Dense(1)

    def call(self, t):
        """
        Forward pass through the network.
        
        Parameters:
            t (tf.Tensor): Input tensor of shape (batch_size, 1) containing time coordinates
            
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, 1) containing alpha2 values
        """
        if self.trainable_alpha2:
            x = t
            for layer in self.hidden_layers:
                x = layer(x)
            # Apply sigmoid to get value between 0 and 1
            sigmoid_output = tf.sigmoid(self.output_layer(x))
            # Scale sigmoid output to be between alpha2_min and alpha2_max
            return self.alpha2_min + (self.alpha2_max - self.alpha2_min) * sigmoid_output
        else:
            batch_size = tf.shape(t)[0]
            return tf.ones((batch_size, 1), dtype=DTYPE) * self.alpha2

    def get_average_alpha2(self, tmin, tmax, num_points=1000):
        """
        Calculate the average alpha2 value over the time domain.
        
        Parameters:
            tmin (float): Minimum time value
            tmax (float): Maximum time value
            num_points (int): Number of points to sample
            
        Returns:
            float: Average alpha2 value
        """
        t_samples = tf.linspace(tmin, tmax, num_points)
        alpha2_values = self.call(tf.expand_dims(t_samples, axis=-1))
        average_alpha2 = tf.reduce_mean(alpha2_values)
        return average_alpha2.numpy()

    def get_alpha2_range(self, tmin, tmax, num_points=1000):
        """
        Calculate the range of alpha2 values over the time domain.
        
        Parameters:
            tmin (float): Minimum time value
            tmax (float): Maximum time value
            num_points (int): Number of points to sample
            
        Returns:
            tuple: (min_alpha2, max_alpha2) Minimum and maximum alpha2 values
        """
        t_samples = tf.linspace(tmin, tmax, num_points)
        alpha2_values = self.call(tf.expand_dims(t_samples, axis=-1))
        min_alpha2 = tf.reduce_min(alpha2_values)
        max_alpha2 = tf.reduce_max(alpha2_values)
        return min_alpha2.numpy(), max_alpha2.numpy()

# =============================================================================
# Latin Hypercube Sampling Implementation
def latin_hypercube_sampling(n_samples, n_dims, bounds, dtype=DTYPE):
    """
    Generate Latin Hypercube samples.
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions
        bounds: List of (min, max) tuples for each dimension
        dtype: Data type for the samples
    
    Returns:
        Array of samples with shape (n_samples, n_dims)
    """
    # Ensure n_samples is a valid integer
    n_samples = int(max(1, n_samples))
    
    # Create the random samples
    samples = np.zeros((n_samples, n_dims))
    
    # Generate samples for each dimension
    for i in range(n_dims):
        # Get bounds for this dimension
        lb, ub = bounds[i]
        # Convert bounds to float if they're tensors
        if isinstance(lb, tf.Tensor):
            lb = lb.numpy()
        if isinstance(ub, tf.Tensor):
            ub = ub.numpy()
            
        # Generate uniform random numbers
        u = np.random.uniform(0, 1, n_samples)
        perm = np.random.permutation(n_samples)
        
        # Generate samples
        samples[:, i] = lb + (ub - lb) * ((perm + u) / n_samples)
    
    return tf.convert_to_tensor(samples, dtype=dtype)

# =============================================================================
# Advanced Adaptive Sampling Methods
def detect_high_gradient_regions(x, v, threshold_factor=0.5):
    """
    Detect regions with high gradients in the solution.
    
    Args:
        x: Input coordinates
        v: Solution values
        threshold_factor: Factor to determine high gradient threshold
    
    Returns:
        Mask indicating high gradient regions, padded to match input size
    """
    # Ensure inputs are 2D tensors
    x = tf.reshape(x, [-1, 1])
    v = tf.reshape(v, [-1, 1])
    
    # Calculate gradients
    gradients = tf.abs(v[1:] - v[:-1]) / tf.abs(x[1:] - x[:-1])
    threshold = tf.reduce_mean(gradients) * threshold_factor
    
    # Create mask for gradients
    grad_mask = gradients > threshold
    
    # Pad the mask to match original size
    paddings = tf.constant([[0, 1], [0, 0]])
    padded_mask = tf.pad(tf.cast(grad_mask, dtype=tf.bool), paddings, constant_values=False)
    
    return padded_mask



def generate_adaptive_samples(model, alpha_net, n_samples, bounds, resolution_levels=3, dtype=DTYPE):
    """
    Generate samples using multi-resolution adaptive sampling.
    
    Args:
        model: Neural network model
        alpha_net: Alpha network
        n_samples: Total number of samples to generate
        bounds: List of (min, max) tuples for each dimension
        resolution_levels: Number of resolution levels
        dtype: Data type for samples
    
    Returns:
        Combined set of adaptive samples
    """
    samples_per_level = n_samples // resolution_levels
    all_samples = []
    
    # Initial coarse sampling using LHS
    coarse_samples = latin_hypercube_sampling(samples_per_level, len(bounds), bounds, dtype)
    all_samples.append(coarse_samples)
    
    if model is not None:
        # Progressive refinement
        for level in range(1, resolution_levels):
            # Evaluate PDE residuals on current samples
            r = residual(model, alpha_net, coarse_samples)
            error_magnitude = tf.abs(r)
            
            # Identify regions needing refinement (fix mask shape)
            high_error_mask = error_magnitude > tf.reduce_mean(error_magnitude)
            high_error_mask = tf.squeeze(high_error_mask, axis=1)  # Remove redundant dimension
            
            high_error_points = tf.boolean_mask(coarse_samples, high_error_mask)
            
            # Generate finer samples around high error regions
            if tf.size(high_error_points) > 0:
                for point in high_error_points[:min(len(high_error_points), 10)]:
                    # Create local bounds around the point
                    local_bounds = []
                    for dim, (lb, ub) in enumerate(bounds):
                        radius = (ub - lb) / (2 ** (level + 1))
                        local_lb = tf.maximum(point[dim] - radius, lb)
                        local_ub = tf.minimum(point[dim] + radius, ub)
                        local_bounds.append((local_lb, local_ub))
                    
                    # Generate local samples
                    local_samples = latin_hypercube_sampling(
                        samples_per_level // len(high_error_points), 
                        len(bounds), 
                        local_bounds, 
                        dtype
                    )
                    all_samples.append(local_samples)
    
    # Combine all samples
    return tf.concat(all_samples, axis=0)

def generate_smart_sampling_points(model, alpha_net, N_r, bounds, initial_condition, dtype=DTYPE):
    """
    Generate training points using multiple sampling strategies.
    """
    # Allocate points to different strategies
    N_lhs = max(1, N_r // 4)        # 25% Latin Hypercube sampling for global coverage
    N_adaptive = max(1, N_r // 4)   # 25% adaptive sampling
    N_shock = max(1, N_r // 4)      # 25% near shocks/discontinuities
    N_boundary = N_r - N_lhs - N_adaptive - N_shock  # 25% near boundaries
    
    samples = []
    
    # 1. Latin Hypercube Sampling for global coverage
    best_lhs = latin_hypercube_sampling(N_lhs, len(bounds), bounds, dtype)
    samples.append(best_lhs)
    
    # 2. Adaptive sampling based on PDE residuals and multi-resolution
    if model is not None:
        adaptive_samples = generate_adaptive_samples(
            model, alpha_net, N_adaptive, bounds, resolution_levels=3, dtype=dtype
        )
        samples.append(adaptive_samples)
    else:
        # If no model available, use additional LHS samples
        samples.append(latin_hypercube_sampling(N_adaptive, len(bounds), bounds, dtype))
    
    # 3. Sampling near shocks and high gradients
    t_dense = tf.cast(tf.linspace(bounds[0][0], bounds[0][1], 100), dtype)
    x_dense = tf.cast(tf.linspace(bounds[1][0], bounds[1][1], 1000), dtype)
    x_dense = tf.reshape(x_dense, [-1, 1])
    
    # Detect high gradient regions in initial condition
    v_initial = initial_condition(x_dense)
    high_grad_mask = detect_high_gradient_regions(x_dense, v_initial)
    
    if tf.reduce_any(high_grad_mask):
        # Sample around high gradient regions
        high_grad_x = tf.boolean_mask(x_dense, high_grad_mask)
        n_grad_points = tf.shape(high_grad_x)[0]
        
        if n_grad_points > 0:
            # Generate samples around each high gradient point
            points_per_region = max(1, N_shock // min(n_grad_points, 10))
            for i in range(min(tf.shape(high_grad_x)[0], 10)):
                x_point = high_grad_x[i]  # This is already a scalar tensor
                # Define local bounds around the point
                local_bounds = [
                    (bounds[0][0], bounds[0][1]),  # Full time range
                    (tf.maximum(x_point - 0.1, bounds[1][0]), 
                     tf.minimum(x_point + 0.1, bounds[1][1]))  # Local x range
                ]
                # Generate local samples
                local_samples = latin_hypercube_sampling(
                    points_per_region, len(bounds), local_bounds, dtype
                )
                samples.append(local_samples)
    
    # If no high gradient regions found or samples list is empty, add more LHS samples
    if len(samples) < 3:
        additional_samples = latin_hypercube_sampling(N_shock, len(bounds), bounds, dtype)
        samples.append(additional_samples)
    
    # 4. Boundary region sampling
    left_bounds = [(bounds[0][0], bounds[0][1]), 
                   (bounds[1][0], bounds[1][0] + 0.1)]
    right_bounds = [(bounds[0][0], bounds[0][1]), 
                    (bounds[1][1] - 0.1, bounds[1][1])]
    
    n_boundary_each = max(1, N_boundary // 2)
    left_samples = latin_hypercube_sampling(n_boundary_each, len(bounds), left_bounds, dtype)
    right_samples = latin_hypercube_sampling(n_boundary_each, len(bounds), right_bounds, dtype)
    
    samples.append(left_samples)
    samples.append(right_samples)
    
    # Combine all samples and ensure correct size
    all_samples = tf.concat(samples, axis=0)
    all_samples = tf.clip_by_value(all_samples, 
                                  [bounds[0][0], bounds[1][0]], 
                                  [bounds[0][1], bounds[1][1]])
    
    # Ensure we have exactly N_r points
    if tf.shape(all_samples)[0] > N_r:
        indices = tf.random.shuffle(tf.range(tf.shape(all_samples)[0]))[:N_r]
        all_samples = tf.gather(all_samples, indices)
    elif tf.shape(all_samples)[0] < N_r:
        extra_samples = latin_hypercube_sampling(
            max(1, N_r - tf.shape(all_samples)[0]), len(bounds), bounds, dtype
        )
        all_samples = tf.concat([all_samples, extra_samples], axis=0)
    
    return all_samples

def generate_data():
    """Generate training data using smart sampling strategies."""
    # Initial condition points
    t_0 = tf.ones((TRAINING_CONFIG['N_0'], 1), dtype=DTYPE) * tf.constant(tmin, dtype=DTYPE)
    x_0_bounds = [(xmin, xmax)]
    x_0 = latin_hypercube_sampling(TRAINING_CONFIG['N_0'], 1, x_0_bounds)
    v_0 = initial_condition(x_0)
    X_0 = tf.concat([t_0, x_0], axis=1)
    
    # Boundary points
    t_b = latin_hypercube_sampling(TRAINING_CONFIG['N_b'], 1, [(tmin, tmax)])
    x_b = tf.constant(xmin, dtype=DTYPE) + (
            tf.constant(xmax, dtype=DTYPE) - tf.constant(xmin, dtype=DTYPE)
          ) * tf.cast(tf.keras.backend.random_bernoulli((TRAINING_CONFIG['N_b'], 1), 0.5), DTYPE)
    bc_result = boundary_condition(t_b, x_b)
    if isinstance(bc_result, tuple):
        v_b_values, v_b_weights = bc_result
        v_b = (v_b_values, v_b_weights)
    else:
        v_b = bc_result
    X_b = tf.concat([t_b, x_b], axis=1)
    
    # Collocation points using smart sampling
    bounds_r = [(tmin, tmax), (xmin, xmax)]
    X_r = generate_smart_sampling_points(None, None, TRAINING_CONFIG['N_r'], 
                                       bounds_r, initial_condition)
    
    return X_0, v_0, X_b, v_b, X_r

def update_collocation_points(model, alpha_net, N_r, N_candidates, lb, ub, DTYPE=DTYPE):
    """Update collocation points during training using adaptive sampling."""
    bounds = [(lb[0], ub[0]), (lb[1], ub[1])]
    return generate_smart_sampling_points(model, alpha_net, N_r, bounds, initial_condition)

# =============================================================================
# Model Initialization
def init_model(num_hidden_layers=8, num_neurons_per_layer=20):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,), dtype=DTYPE))
    model.add(tf.keras.layers.Lambda(lambda x: tf.cast(2.0, dtype=DTYPE) * (x - lb) / (ub - lb) - tf.cast(1.0, dtype=DTYPE),
                                     dtype=DTYPE))
    initializer = tf.keras.initializers.GlorotNormal(seed=1234)
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        activation='tanh',
                                        kernel_initializer=initializer,
                                        dtype=DTYPE))
    model.add(tf.keras.layers.Dense(1, dtype=DTYPE))
    model.build((None, 2))
    model.summary()
    # Force-cast weights to the desired dtype if necessary.
    if DTYPE == "float64":
        new_weights = [tf.cast(w, tf.float64) for w in model.get_weights()]
        model.set_weights(new_weights)
    return model

# =============================================================================
# PDE Residual
# This part comes from "lb_inf_grok_v3.py"
def residual(model, alpha_net, X_r):
        t, x = X_r[:, 0:1], X_r[:, 1:2]
        with tf.GradientTape() as tape2:
            tape2.watch([t, x])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([t, x])
                v = model(tf.concat([t, x], axis=1))
            v_t = tape1.gradient(v, t)
            v_x = tape1.gradient(v, x)
            del tape1
            if v_t is None or v_x is None:
                raise ValueError("Gradients v_t or v_x are None.")
        v_xx = tape2.gradient(v_x, x)
        if v_xx is None:
            raise ValueError("Second derivative v_xx is None.")
        alpha2 = alpha_net(t)
        return v_t + v * v_x + alpha2 * v_x * v_xx 


# =============================================================================
# Dynamic Loss Weights
class DynamicWeights:
    def __init__(self, initial_weights={'residual': 1.0, 'initial': 1.0, 'boundary': 1.0},
                 update_frequency=1000, smoothing_factor=0.9):
        # Convert initial weights to the correct dtype
        self.weights = {
            key: tf.cast(value, dtype=DTYPE) 
            for key, value in initial_weights.items()
        }
        self.update_frequency = update_frequency
        self.smoothing_factor = tf.cast(smoothing_factor, dtype=DTYPE)
        self.loss_history = {'residual': [], 'initial': [], 'boundary': []}
        self.epoch_counter = 0
        
    def update(self, current_losses):
        """Update weights based on relative magnitudes of losses"""
        self.epoch_counter += 1
        
        # Store current losses
        for key, value in current_losses.items():
            self.loss_history[key].append(tf.cast(value, dtype=DTYPE))
        
        # Update weights every update_frequency epochs
        if self.epoch_counter % self.update_frequency == 0:
            # Compute mean of recent losses
            recent_means = {
                key: tf.reduce_mean(tf.cast(self.loss_history[key][-self.update_frequency:], dtype=DTYPE))
                for key in self.loss_history
            }
            
            # Compute inverse weights (smaller loss -> larger weight)
            total_loss = tf.reduce_sum(list(recent_means.values()))
            base_weights = {
                key: total_loss / (value + tf.cast(1e-10, dtype=DTYPE))
                for key, value in recent_means.items()
            }
            
            # Normalize weights to sum to 3.0
            total_weight = tf.reduce_sum(list(base_weights.values()))
            new_weights = {
                key: (value / total_weight) * tf.cast(3.0, dtype=DTYPE)
                for key, value in base_weights.items()
            }
            
            # Apply smoothing
            self.weights = {
                key: self.smoothing_factor * self.weights[key] + 
                     (tf.cast(1.0, dtype=DTYPE) - self.smoothing_factor) * new_weights[key]
                for key in self.weights
            }
            
            # Print weight update information for debugging
           # print("\nWeight Update Information:")
           # print(f"Recent mean losses: {recent_means}")
           # print(f"New weights: {new_weights}")
           # print(f"Smoothed weights: {self.weights}\n")
            
            # Clear history after update
            self.loss_history = {key: [] for key in self.loss_history}
        
        return self.weights

# =============================================================================
# Loss Computation with Dynamic Weights
def compute_loss(model, alpha_net, X_r, X_0, v_0, X_b, v_b, dynamic_weights=None):
    with tf.GradientTape(persistent=True) as tape:
        r = residual(model, alpha_net, X_r)
        loss_residual = tf.reduce_mean(tf.square(r))
        
        v_0_pred = model(X_0)
        loss_initial = tf.reduce_mean(tf.square(v_0 - v_0_pred))
        
        v_b_pred = model(X_b)
        # Handle boundary conditions that return (values, weights)
        if isinstance(v_b, tuple):
            v_b_values, v_b_weights = v_b
            loss_boundary = tf.reduce_mean(tf.square(v_b_values - v_b_pred) * v_b_weights)
        else:
            loss_boundary = tf.reduce_mean(tf.square(v_b - v_b_pred))
        
        # Get current weights
        if dynamic_weights is not None:
            weights = dynamic_weights.update({
                'residual': loss_residual.numpy(),
                'initial': loss_initial.numpy(),
                'boundary': loss_boundary.numpy()
            })
        else:
            weights = {'residual': 1.0, 'initial': 1.0, 'boundary': 1.0}
        
        # Apply weights to losses
        total_loss = (weights['residual'] * loss_residual + 
                     weights['initial'] * loss_initial + 
                     weights['boundary'] * loss_boundary)
    
    return loss_residual, loss_initial, loss_boundary, total_loss, weights

# =============================================================================
# Training Step with Dynamic Weights
@tf.function
def train_step(model, alpha_net, X_r, X_0, v_0, X_b, v_b, weights, model_optimizer, alpha_optimizer):
    """Perform one training step."""
    with tf.GradientTape(persistent=True) as tape:
        # Compute losses
        loss_residual = tf.reduce_mean(tf.square(residual(model, alpha_net, X_r)))
        loss_initial = tf.reduce_mean(tf.square(model(X_0) - v_0))
        loss_boundary = tf.reduce_mean(tf.square(model(X_b) - v_b))
        
        # Compute total loss with weights
        total_loss = (weights['residual'] * loss_residual + 
                     weights['initial'] * loss_initial + 
                     weights['boundary'] * loss_boundary)
    
    # Compute and apply gradients
    model_gradients = tape.gradient(total_loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    
    if not FIX_ALPHA2:
        alpha_gradients = tape.gradient(total_loss, alpha_net.trainable_variables)
        alpha_optimizer.apply_gradients(zip(alpha_gradients, alpha_net.trainable_variables))
    
    del tape
    
    return loss_residual, loss_initial, loss_boundary, total_loss

# =============================================================================
# Computing L2 Error over time
def compute_l2_error(model, IB, xmin, xmax, tmin, tmax, Nx_test_grid, Nt_test_grid, DTYPE):
    """
    Compute relative L2 errors between model predictions and true solution.
    
    Parameters:
        model (tf.keras.Model): Neural network model for the solution
        IB (int): Initial/boundary condition index
        xmin (float): Minimum spatial value
        xmax (float): Maximum spatial value
        tmin (float): Minimum time value
        tmax (float): Maximum time value
        Nx_test_grid (int): Number of spatial points for testing
        Nt_test_grid (int): Number of time points for testing
        DTYPE: Data type for TensorFlow operations
        
    Returns:
        tuple: (final_l2_error, avg_l2_error, X_test, x_test, t_test, u_pred, u_exact)
            final_l2_error (tf.Tensor): Relative L2 error at final time
            avg_l2_error (tf.Tensor): Average relative L2 error over all time steps
            X_test (tf.Tensor): Test grid points
            x_test (tf.Tensor): x coordinates
            t_test (tf.Tensor): t coordinates
            u_pred (tf.Tensor): Model predictions
            u_exact (tf.Tensor): True solution values
    """
    # Generate test grids
    x_test = tf.linspace(tf.constant(xmin, dtype=DTYPE),
                         tf.constant(xmax, dtype=DTYPE), Nx_test_grid)
    t_test = tf.linspace(tf.constant(tmin, dtype=DTYPE),
                         tf.constant(tmax, dtype=DTYPE), Nt_test_grid)
    
    # Calculate grid spacing
    dx = (xmax - xmin) / (Nx_test_grid - 1)
    spatial_domain_size = xmax - xmin
    
    X, T = tf.meshgrid(x_test, t_test, indexing='xy')
    X_test = tf.reshape(tf.stack([T, X], axis=2), [-1, 2])
    
    # Get true solution function
    u_true = get_true_solution(IB)
    
    # Compute predictions and true values
    u_pred = model(X_test)
    u_exact = u_true(X_test)
    
    # Reshape predictions and exact values
    u_pred_reshaped = tf.reshape(u_pred, [Nt_test_grid, Nx_test_grid])
    u_exact_reshaped = tf.reshape(u_exact, [Nt_test_grid, Nx_test_grid])
    
    # Compute L2 error for each time step
    squared_diff = tf.square(u_exact_reshaped - u_pred_reshaped)
    squared_exact = tf.square(u_exact_reshaped)
    
    # Compute absolute L2 error (numerator)
    error_integral = tf.reduce_sum(squared_diff, axis=1) * dx
    absolute_l2_errors = tf.sqrt(error_integral / spatial_domain_size)
    
    # Compute norm of exact solution (denominator)
    exact_integral = tf.reduce_sum(squared_exact, axis=1) * dx
    exact_l2_norms = tf.sqrt(exact_integral / spatial_domain_size)
    
    # Compute relative L2 error
    # Add small epsilon to avoid division by zero
    epsilon = tf.constant(1e-10, dtype=DTYPE)
    relative_l2_errors = absolute_l2_errors / (exact_l2_norms + epsilon)
    
    # Compute final and average relative L2 errors
    final_l2_error = relative_l2_errors[-1]
    avg_l2_error = tf.reduce_mean(relative_l2_errors)
    
    return final_l2_error, avg_l2_error, X_test, x_test, t_test, u_pred, u_exact

# =============================================================================
# Training Loop with Optional Dynamic Weights
def train_model(model, alpha_net, X_r, X_0, v_0, X_b, v_b, metrics_filename, epochs=10000, domain_bounds=None):
    """
    Train the model.
    
    Performs the training loop, computes losses, updates parameters, and records metrics.
    
    Parameters:
        model (tf.keras.Model): Neural network model for the solution
        alpha_net (Alpha2Net): Neural network model for alpha2
        X_r (tf.Tensor): Residual points
        X_0 (tf.Tensor): Initial condition points
        v_0 (tf.Tensor): Initial condition values
        X_b (tf.Tensor): Boundary condition points
        v_b (tf.Tensor): Boundary condition values
        metrics_filename (str): Filename to save training metrics
        epochs (int): Number of training epochs
        domain_bounds (tuple, optional): Domain bounds (xmin, xmax)
        
    Returns:
        dict: Training history containing losses and metrics
    """
    try:
        # Record start time
        start_time = time()
        
        # Initialize optimizers
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Initialize weights
        if TRAINING_CONFIG['USE_DYNAMIC_WEIGHTS']:
            dynamic_weights = DynamicWeights(
                initial_weights=TRAINING_CONFIG['initial_weights'],
                update_frequency=TRAINING_CONFIG['weight_update_frequency'],
                smoothing_factor=TRAINING_CONFIG['weight_smoothing_factor']
            )
            current_weights = dynamic_weights.weights
        else:
            current_weights = TRAINING_CONFIG['initial_weights']
        
        # Initialize history
        history = {
            'loss_residual': [],
            'loss_initial': [],
            'loss_boundary': [],
            'total_loss': [],
            'alpha2_values': [],
            'l2_error_values': [],
            'avg_l2_error_values': [],  # Added average L2 error tracking
            'training_time': 0.0
        }
        
        # Extract domain bounds if provided, otherwise use global variables
        if domain_bounds:
            xmin, xmax = domain_bounds
        else:
            # Use global variables
            xmin, xmax = globals()['xmin'], globals()['xmax']
            
        # Training loop
        for epoch in range(epochs + 1):
            # Training step
            loss_residual, loss_initial, loss_boundary, total_loss = train_step(
                model=model,
                alpha_net=alpha_net,
                X_r=X_r,
                X_0=X_0,
                v_0=v_0,
                X_b=X_b,
                v_b=v_b,
                weights=current_weights,
                model_optimizer=model_optimizer,
                alpha_optimizer=alpha_optimizer
            )
            
            # Update weights if using dynamic weighting
            if TRAINING_CONFIG['USE_DYNAMIC_WEIGHTS']:
                current_weights = dynamic_weights.update({
                    'residual': float(loss_residual),
                    'initial': float(loss_initial),
                    'boundary': float(loss_boundary)
                })
            
            # Store training history
            history['loss_residual'].append(float(loss_residual))
            history['loss_initial'].append(float(loss_initial))
            history['loss_boundary'].append(float(loss_boundary))
            history['total_loss'].append(float(total_loss))
            
            # Get current alpha2 value (handle both fixed and trainable cases)
            if not alpha_net.trainable_alpha2:
                current_alpha2 = float(alpha_net.alpha2.numpy())
            else:
                # For trainable alpha2, get the average value
                current_alpha2 = float(alpha_net.get_average_alpha2(globals()['tmin'], globals()['tmax']))
            
            history['alpha2_values'].append(current_alpha2)
            
            # Compute L2 error every 1000 epochs
            if epoch % 1000 == 0:
                final_l2_error, avg_l2_error, _, _, _, _, _ = compute_l2_error(
                    model, IB, xmin, xmax, globals()['tmin'], globals()['tmax'], 
                    globals()['Nx_test_grid'], globals()['Nt_test_grid'], DTYPE
                )
                history['l2_error_values'].append(float(final_l2_error.numpy()))
                history['avg_l2_error_values'].append(float(avg_l2_error.numpy()))
                
                # Log progress
                if not alpha_net.trainable_alpha2:
                    print(f"\nEpoch {epoch:05d}, Total Loss: {total_loss:.4e}, alpha2: {current_alpha2:.6f}")
                else:
                    min_alpha2, max_alpha2 = alpha_net.get_alpha2_range(globals()['tmin'], globals()['tmax'])
                    print(f"\nEpoch {epoch:05d}, Total Loss: {total_loss:.4e}, alpha2 range: [{min_alpha2:.6f}, {max_alpha2:.6f}]")
                
                print(f"Final L2 Error: {final_l2_error:.4e}, Average L2 Error: {avg_l2_error:.4e}")
        
        # Calculate training time
        training_time = time() - start_time
        print(f"Training completed successfully in {training_time:.2f} seconds")
        
        # Add training time to history
        history['training_time'] = training_time
        
        # Get final alpha2 value
        if not FIX_ALPHA2:
            final_alpha2 = alpha_net.get_average_alpha2(globals()['tmin'], globals()['tmax'])
        else:
            final_alpha2 = alpha_net.alpha2.numpy()
        sqrt_alpha2_value = np.sqrt(final_alpha2)
        
        # Write final metrics
        with open(metrics_filename, 'w') as f:
            metrics = []  # Store all lines in a list
            metrics.append(f"Final Training Metrics for IB = {IB}")
            metrics.append("================================================================================")
            metrics.append(f"Total Epochs: {epochs}")
            metrics.append(f"Dynamic Weights: {TRAINING_CONFIG['USE_DYNAMIC_WEIGHTS']}")
            metrics.append(f"Final Total Loss: {total_loss:.4e}")
            metrics.append(f"Final alpha2: {final_alpha2:.6f}")
            metrics.append(f"Final sqrt(alpha2): {sqrt_alpha2_value:.6f}")
            metrics.append(f"Final L2 Error: {final_l2_error:.4e}")
            metrics.append(f"Average L2 Error: {avg_l2_error:.4e}")
            metrics.append("Final Weights:")
            
            # Add weights information
            if TRAINING_CONFIG['USE_DYNAMIC_WEIGHTS']:
                current_weights = dynamic_weights.weights
                for key in ['residual', 'initial', 'boundary']:  # Fixed order
                    value = float(current_weights[key]) if isinstance(current_weights[key], tf.Tensor) else current_weights[key]
                    metrics.append(f"  {key}: {value:.6f}")
            else:
                for key in ['residual', 'initial', 'boundary']:  # Fixed order
                    metrics.append(f"  {key}: {TRAINING_CONFIG['initial_weights'][key]:.6f}")
            
            metrics.append(f"Training Time: {training_time:.2f} seconds")
            
            # Write all lines with no extra newlines
            f.write('\n'.join(metrics))
        
        return history
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# Plot Training History with Weights
def plot_training_history(history, filename):
    """
    Plot training history including losses and weights if available.
    
    Parameters:
        history (dict): Training history containing losses and metrics
        filename (str): Filename to save the plot
    """
    # Check if history is None
    if history is None:
        # Create a simple figure with an error message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Training failed. No history data available.", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    axs[0, 0].semilogy(history['total_loss'], label='Total Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss (log scale)')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Plot component losses
    axs[0, 1].semilogy(history['loss_residual'], label='Residual')
    axs[0, 1].semilogy(history['loss_initial'], label='Initial')
    axs[0, 1].semilogy(history['loss_boundary'], label='Boundary')
    axs[0, 1].set_xlabel('Epoch', fontsize=14)
    axs[0, 1].set_ylabel('Loss (log scale)', fontsize=14)
    axs[0, 1].set_title('Component Losses', fontsize=14)
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Plot alpha2 values
    axs[1, 0].plot(history['alpha2_values'], label='Alpha2')
    axs[1, 0].set_xlabel('Epoch', fontsize=14)
    axs[1, 0].set_ylabel('Alpha2', fontsize=14)
    axs[1, 0].set_title('Alpha2 Value', fontsize=14)
    axs[1, 0].grid(True)
    
    # Plot weights if available
    if 'weights' in history:
        for key in history['weights']:
            axs[1, 1].plot(history['weights'][key], label=f'{key.capitalize()} Weight')
        axs[1, 1].set_xlabel('Epoch', fontsize=14)
        axs[1, 1].set_ylabel('Weight', fontsize=14)
        axs[1, 1].set_title('Loss Weights', fontsize=14)
        axs[1, 1].grid(True)
        axs[1, 1].legend()
    else:
        # If weights are not available, display a message
        axs[1, 1].text(0.5, 0.5, 'Weights not available\n(Dynamic weights disabled)', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('Loss Weights')
        axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# Plot Error Analysis
def plot_error_analysis(error_metrics, x_test, t_test, filename):
    """
    Create comprehensive error analysis visualization.
    """
    plt.figure(figsize=(20, 12))
    
    # 1. Temporal evolution of errors
    plt.subplot(2, 3, 1)
    plt.plot(t_test, error_metrics['l2_errors_per_time'], 'b-', label='L2 Error')
    plt.plot(t_test, error_metrics['temporal_max_error'], 'r--', label='Max Error')
    plt.plot(t_test, error_metrics['temporal_mean_error'], 'g:', label='Mean Error')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Error Magnitude', fontsize=14)
    plt.title('Temporal Evolution of Errors', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # 2. Spatial distribution of errors
    plt.subplot(2, 3, 2)
    plt.plot(x_test, error_metrics['spatial_max_error'], 'r-', label='Max Error')
    plt.plot(x_test, error_metrics['spatial_mean_error'], 'b--', label='Mean Error')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('Error Magnitude', fontsize=14)
    plt.title('Spatial Distribution of Errors', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # 3. Error histogram
    plt.subplot(2, 3, 3)
    all_errors = np.concatenate([error_metrics['temporal_mean_error'], 
                               error_metrics['spatial_mean_error']])
    plt.hist(all_errors, bins=50, density=True, alpha=0.7)
    plt.xlabel('Error Magnitude', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Error Distribution', fontsize=14)
    plt.grid(True)
    
    # 4. Box plot of temporal errors
    plt.subplot(2, 3, 4)
    plt.boxplot(error_metrics['l2_errors_per_time'])
    plt.ylabel('L2 Error Magnitude', fontsize=14)
    plt.title('Distribution of L2 Errors Over Time', fontsize=14)
    plt.grid(True)
    
    # 5. Error statistics table
    plt.subplot(2, 3, 5)
    plt.axis('off')
    stats_text = f"""Error Statistics:
    Mean Error: {error_metrics['mean_error']:.2e}
    Median Error: {error_metrics['median_error']:.2e}
    Max Error: {error_metrics['max_error']:.2e}
    Std Dev: {error_metrics['std_error']:.2e}
    25th Percentile: {error_metrics['percentiles']['25']:.2e}
    75th Percentile: {error_metrics['percentiles']['75']:.2e}"""
    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
# ============================================================================
# File name generation
def generate_filenames(result_dir, ib, alpha_str, epochs):
    """
    Generate filenames for saving results.
    
    Parameters:
        result_dir (str): Directory to save results
        ib (int): Initial/boundary condition index
        alpha_str (str): String representation of alpha value
        epochs (int): Number of training epochs
    
    Returns:
        tuple: (solution_filename, loss_error_filename, heatmap_filename, metrics_filename)
    """
    # Create base filenames
    base_filename1 = os.path.join(result_dir, f"IB{ib}_solution_{alpha_str}_{epochs}")
    base_filename2 = os.path.join(result_dir, f"IB{ib}_loss_error_{alpha_str}_{epochs}")
    base_filename3 = os.path.join(result_dir, f"IB{ib}_loss_heatmap_{alpha_str}_{epochs}")
    base_filename4 = os.path.join(result_dir, f"IB{ib}_metrics_{alpha_str}_{epochs}")

    # Find the next available filename
    counter1, counter2, counter3, counter4 = 0, 0, 0, 0
    
    while os.path.exists(f"{base_filename1}_{counter1}.png"):
        counter1 += 1
    while os.path.exists(f"{base_filename2}_{counter2}.png"):
        counter2 += 1
    while os.path.exists(f"{base_filename3}_{counter3}.png"):
        counter3 += 1
    while os.path.exists(f"{base_filename4}_{counter4}.png") or os.path.exists(f"{base_filename4}_{counter4}.txt"):
        counter4 += 1

    # Set the final filenames with counters
    filename1 = f"{base_filename1}_{counter1}.png"
    filename2 = f"{base_filename2}_{counter2}.png"
    filename3 = f"{base_filename3}_{counter3}.png"
    metrics_filename = f"{base_filename4}_{counter4}.txt"
    
    return filename1, filename2, filename3, metrics_filename
# =============================================================================
# Main execution
if __name__ == "__main__":
    # Create temporary metrics file in results directory
    temp_metrics_file = os.path.join(RESULT_DIR, "temp.txt")
    
    # Load exact data to get domain bounds
    exact_t, exact_x, exact_v, tmin, tmax, xmin, xmax, Nt_test_grid, Nx_test_grid = load_exact_data()
    
    # Set domain bounds
    lb = tf.constant([tmin, xmin], dtype=DTYPE)
    ub = tf.constant([tmax, xmax], dtype=DTYPE)
    
    # Get initial and boundary conditions FIRST (before generate_data)
    initial_condition, boundary_condition = get_initial_boundary_conditions(IB, xmin, xmax, DTYPE, pi)
    
    # Initialize alpha2 network
    if FIX_ALPHA2:
        alpha_net = Alpha2Net(trainable_alpha2=False, initial_value=ALPHA2_VALUE)
        print(f"Using fixed alpha2 = {ALPHA2_VALUE:.6f} (alpha = {ALPHA:.6f})")
    else:
        alpha_net = Alpha2Net(trainable_alpha2=True, initial_value=ALPHA2_VALUE)
        print(f"Using trainable alpha2(t) with initial value {ALPHA2_VALUE:.6f} (alpha = {ALPHA:.6f})")
    
    # Initialize neural network model
    model = init_model(
        num_hidden_layers=TRAINING_CONFIG['num_hidden_layers'],
        num_neurons_per_layer=TRAINING_CONFIG['num_neurons_per_layer']
    )
    
    # Generate training data - capture all returned values
    data = generate_data()
    
    # Unpack the data based on what generate_data actually returns
    # This is a guess - adjust based on the actual return values of generate_data
    X_0, v_0, X_b, v_b, X_r = data
    
    # Initial alpha string for filenames (will be updated after training)
    initial_sqrt_alpha2_str = f"{np.sqrt(alpha_net.alpha2.numpy()):.4f}"
    
    # Create alpha string for filenames
    if FIX_ALPHA2:
        initial_alpha_str = f"alpha_{initial_sqrt_alpha2_str}"
    else:
        initial_alpha_str = f"alpha(t){initial_sqrt_alpha2_str}"
    
    # Generate initial filenames
    filename1, filename2, filename3, metrics_filename = generate_filenames(
        RESULT_DIR, IB, initial_alpha_str, TRAINING_CONFIG['N_epoch']
    )
    
    # Train the model and write metrics to the temporary file
    history = train_model(model, alpha_net, X_r, X_0, v_0, X_b, v_b, temp_metrics_file, 
                         epochs=TRAINING_CONFIG['N_epoch'])
    
    # Check if training was successful
    if history is None:
        print("Training failed. Cannot continue with visualization and analysis.")
        # Create a basic metrics file with error information
        with open(metrics_filename, 'w') as f:
            f.write(f"Final Training Metrics for IB = {IB}\n")
            f.write("=" * 80 + "\n")
            f.write("Training failed. See console output for error details.\n")
        exit(1)  # Exit with error code
    
    # Calculate final alpha2 value
    if not FIX_ALPHA2:
        # For trainable alpha2, get the average value
        avg_alpha2 = alpha_net.get_average_alpha2(tmin, tmax)
        min_alpha2, max_alpha2 = alpha_net.get_alpha2_range(tmin, tmax)
        print(f"\nFinal alpha2 statistics:")
        print(f"Average alpha2: {avg_alpha2:.6f}")
        print(f"Range: [{min_alpha2:.6f}, {max_alpha2:.6f}]")
        final_alpha2 = avg_alpha2  # Use average as the final value
    else:
        # For fixed alpha2, use the constant value
        final_alpha2 = alpha_net.alpha2.numpy()
    
    sqrt_alpha2_value = np.sqrt(final_alpha2)
    
    # Create final alpha string for filenames
    if FIX_ALPHA2:
        final_alpha_str = f"alpha_{sqrt_alpha2_value:.4f}"
    else:
        final_alpha_str = f"alpha(t){sqrt_alpha2_value:.4f}"
    
    # Generate final filenames with updated alpha value
    filename1, filename2, filename3, metrics_filename = generate_filenames(
        RESULT_DIR, IB, final_alpha_str, TRAINING_CONFIG['N_epoch']
    )
    
    # Compute final L2 error
    final_l2_error, avg_l2_error, X_test, x_test, t_test, u_pred, u_exact = compute_l2_error(
        model, IB, xmin, xmax, tmin, tmax, Nx_test_grid, Nt_test_grid, DTYPE
    )
    
    # Create metrics file
    if os.path.exists(temp_metrics_file):
        os.rename(temp_metrics_file, metrics_filename)
    else:
        print(f"Warning: Temporary metrics file {temp_metrics_file} not found. Creating new metrics file.")
        create_metrics_file(metrics_filename, history, IB, final_alpha2, sqrt_alpha2_value, 
                           final_l2_error, avg_l2_error)
    
    # Plot Solutions at Different Time Slots
    Nt_grid = Nt_test_grid
    t_slots = [0, int(np.floor((Nt_grid+1)/4)), int(np.floor(2*(Nt_grid+1)/4)), Nt_grid-1]
    t_test_np = t_test.numpy()
    t_values = [0, t_test_np[t_slots[1]].item(), t_test_np[t_slots[2]].item(), t_test_np[t_slots[3]].item()]
    u_true = get_true_solution(IB)
    
    plt.figure(figsize=(15, 4))
    for i, (t_idx, t_val) in enumerate(zip(t_slots, t_values)):
        plt.subplot(1, 4, i+1)
        x_plot = np.linspace(xmin-0.1, xmax, 1000, dtype=np.float64)
        t_plot = np.ones_like(x_plot) * t_val
        X_plot = np.stack([t_plot, x_plot], axis=1)
        X_plot_tf = tf.convert_to_tensor(X_plot, dtype=DTYPE)
        u_pred_plot = model(X_plot_tf).numpy().flatten()
        u_true_plot = u_true(X_plot_tf).numpy().flatten()
        print(f"For t={t_val:.2f}, u_true_plot min: {np.min(u_true_plot)}, max: {np.max(u_true_plot)}, any NaN: {np.any(np.isnan(u_true_plot))}") #For debugging
        plt.plot(x_plot, u_pred_plot, 'ro', fillstyle='none', markersize=3, label='Predicted')
        plt.plot(x_plot, u_true_plot, 'b--', linewidth=2, label='True')
        plt.xlabel(r'\textbf{x}', fontsize=18)
        plt.ylabel(r'\textbf{v(x,t)}', fontsize=18)
        plt.title(r'\textbf{t = ' + f'{t_val:.2f}' + '}', fontsize=20)
        # Explicitly set tick label size
        plt.tick_params(axis='both', labelsize=16)  # Matches your global default, or 
                                                    # increase to 16
        plt.xticks(np.arange(int(xmin), int(xmax)+1, 1))
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(filename1, dpi=600, bbox_inches='tight')
    plt.show()

    # Plot training history
    plot_training_history(history, filename2)

    # Plot solution comparison if exact solution is available
    exact_t, exact_x, exact_v, tmin, tmax, xmin, xmax, Nt_test_grid, Nx_test_grid = load_exact_data()
    if exact_v is not None:
        u_pred_reshaped = u_pred.numpy().reshape(Nt_test_grid, Nx_test_grid)
        t_ticks = np.linspace(0, tmax, 5)
        x_ticks = np.linspace(xmin, xmax, 6)
        t_positions = np.linspace(0, Nt_test_grid, 5)
        x_positions = np.linspace(0, Nx_test_grid, 6)
    
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        im1 = ax1.pcolor(exact_v, cmap='viridis')
        ax1.set_title('Exact Solution', fontsize=20)
        ax1.set_xlabel('x', fontsize=18)
        ax1.set_ylabel('t', fontsize=18)
        ax1.set_xticks(x_positions)
        ax1.set_yticks(t_positions)
        ax1.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=18)
        ax1.set_yticklabels([f'{t:.1f}' for t in t_ticks], fontsize=18)
        plt.colorbar(im1, ax=ax1)
    
        im2 = ax2.pcolor(u_pred_reshaped, cmap='viridis')
        ax2.set_title('Predicted Solution', fontsize=20)
        ax2.set_xlabel('x', fontsize=18)
        ax2.set_ylabel('t', fontsize=18)
        ax2.set_xticks(x_positions)
        ax2.set_yticks(t_positions)
        ax2.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=18)
        ax2.set_yticklabels([f'{t:.1f}' for t in t_ticks], fontsize=18)
        plt.colorbar(im2, ax=ax2)
    
        difference = exact_v - u_pred_reshaped
        im3 = ax3.pcolor(difference, cmap='viridis')
        ax3.set_title('Difference (Exact - Predicted)', fontsize=18)
        ax3.set_xlabel('x', fontsize=16)
        ax3.set_ylabel('t', fontsize=16)
        ax3.set_xticks(x_positions)
        ax3.set_yticks(t_positions)
        ax3.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=16)
        ax3.set_yticklabels([f'{t:.1f}' for t in t_ticks], fontsize=16)
        plt.colorbar(im3, ax=ax3)
    
        plt.tight_layout()
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Exact solution data not found; skipping error distribution plot.")

    # After training, add this code to get alpha2 statistics
    if not FIX_ALPHA2:
        avg_alpha2 = alpha_net.get_average_alpha2(tmin, tmax)
        min_alpha2, max_alpha2 = alpha_net.get_alpha2_range(tmin, tmax)
        print(f"\nFinal alpha2 statistics:")
        print(f"Average alpha2: {avg_alpha2:.6f}")
        print(f"Range: [{min_alpha2:.6f}, {max_alpha2:.6f}]")
        
        # Update filename generation to include alpha2 range
        if FIX_ALPHA2:
            # For fixed alpha2, use single value
            sqrt_alpha2_str = f"{np.sqrt(final_alpha2):.4f}"
            alpha_str = f"alpha_{sqrt_alpha2_str}"
        else:
            # For time-dependent alpha2, use range in filename
            alpha_str = f"alpha(t)_{np.sqrt(min_alpha2):.4f}_{np.sqrt(max_alpha2):.4f}"

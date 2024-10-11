import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import cpu_count
from time import time

# For better resolution in plot (png format)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# For better font
plt.rc('font', family='serif', size=12)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=12)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

# Configure GPUs
def configure_gpus(selected_gpu_indices=[0, 1]):
    all_gpus = tf.config.list_physical_devices('GPU')
    if all_gpus:
        try:
            selected_gpus = [all_gpus[i] for i in selected_gpu_indices]
            tf.config.set_visible_devices(selected_gpus, 'GPU')
            strategy = tf.distribute.MirroredStrategy()
            num_gpus = strategy.num_replicas_in_sync
            print(f"Number of GPUs being used: {num_gpus}")
            for i, gpu in enumerate(selected_gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                device_name = gpu_details.get('device_name', gpu.name)
                print(f"GPU {selected_gpu_indices[i]}: {device_name}")
            return strategy
        except RuntimeError as e:
            print(f"Error initializing GPUs: {e}")
    else:
        print("CUDA is not available. Running on CPU.")
        return None

# Set global configuration
def set_global_config(dtype='float64', seed=1234):
    tf.keras.backend.set_floatx(dtype)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Initialize constants
def initialize_constants():
    pi = tf.constant(np.pi, dtype=tf.keras.backend.floatx())
    tmin, tmax, xmin, xmax = 0.0, 4.0, -2.0, 4.0
    lb = tf.constant([tmin, xmin], dtype=tf.keras.backend.floatx())
    ub = tf.constant([tmax, xmax], dtype=tf.keras.backend.floatx())
    return pi, lb, ub

# Data loader for exact solution data
def load_exact_data():
    exact_t = np.load('./data/exact_data4_t.npy')
    exact_x = np.load('./data/exact_data4_x.npy')
    exact_v = np.load('./data/exact_data4_v.npy')
    exact_t_long = np.load('./data/exact_data4_t_time12.npy')
    exact_x_long = np.load('./data/exact_data4_x_time12.npy')
    exact_v_long = np.load('./data/exact_data4_v_time12.npy')
    return exact_t, exact_x, exact_v, exact_t_long, exact_x_long, exact_v_long

# Define the initial condition function
def fun_v_0(x):
    cond1 = tf.cast(tf.math.less(x, 0.0), dtype=tf.float64)
    cond2 = tf.cast(tf.math.logical_and(tf.math.less_equal(x, 1.0), tf.math.greater_equal(x, 0.0)), dtype=tf.float64)
    cond3 = tf.cast(tf.math.greater(x, 1.0), dtype=tf.float64)
    a = tf.math.multiply(cond1, 0.0)
    b = tf.math.multiply(cond2, 1.0)
    c = tf.math.multiply(cond3, 0.0)
    return a + b + c

# Initialize model (MLP)
def init_model(lb, ub, num_hidden_layers=8, num_neurons_per_layer=20):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)))
    
    # Scaling layer to map input to [-1, 1]
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
    model.add(scaling_layer)
    
    # Add hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, activation='tanh', kernel_initializer='glorot_normal'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    
    return model

# PDE residual calculation
def fun_r(t, x, v, v_t, v_x, v_xx, alpha2):
    return v_t + v * v_x + alpha2 * v_x * v_xx

# Function to calculate PDE residuals
def get_r(model, alpha2, X_r):
    with tf.GradientTape(persistent=True) as tape:
        t, x = X_r[:, 0:1], X_r[:, 1:2]
        tape.watch([t, x])

        # Model output for v(t, x)
        v = model(tf.stack([t[:, 0], x[:, 0]], axis=1))

        # First derivatives
        v_x = tape.gradient(v, x)
    v_t = tape.gradient(v, t)
    v_xx = tape.gradient(v_x, x)
    
    del tape
    return fun_r(t, x, v, v_t, v_x, v_xx, alpha2)

# Gradient calculation
def get_grad(model, alpha2, X_r, X_data, v_data):
    with tf.GradientTape(persistent=True) as tape:
        # Watch trainable variables and alpha2
#        trainable_vars = model.trainable_variables + [alpha2]
        tape.watch(model.trainable_variables)

        # Compute the loss
        loss_ics_value, loss_bry_value, loss_res_value, loss = compute_loss(model, alpha2, X_r, X_data, v_data)

    # Compute the gradients
    grad_theta = tape.gradient(loss, model.trainable_variables + [alpha2])

    del tape  # Cleanup GradientTape

    return loss_ics_value, loss_bry_value, loss_res_value, loss, grad_theta

# Compute loss
def compute_loss(model, alpha2, X_r, X_data, v_data):
    w_ics, w_bry, w_res = 1.0, 1.0, 1.0
    r = get_r(model, alpha2, X_r)
    phi_r = tf.reduce_mean(tf.square(r))
    
    v_pred_ics, v_true_ics = model(X_data[0]), v_data[0]
    loss_ics_value = w_ics * tf.reduce_mean(tf.square(v_true_ics - v_pred_ics))

    v_pred_bry, v_true_bry = model(X_data[1]), v_data[1]
    loss_bry_value = w_bry * tf.reduce_mean(tf.square(v_true_bry - v_pred_bry))
    
    loss_res_value = w_res * phi_r
    loss = loss_res_value + loss_ics_value + loss_bry_value

    return loss_ics_value, loss_bry_value, loss_res_value, loss

@tf.function
def train_step(model, alpha2, X_r, X_data, v_data, optim):
    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Compute loss and gradients
        loss_ics_value, loss_bry_value, loss_res_value, loss, grad_theta = get_grad(model, alpha2, X_r, X_data, v_data)

    # Combine trainable variables and alpha2
  #  trainable_vars = model.trainable_variables + [alpha2]

    # Ensure gradients are not None
    grad_theta = [g for g in grad_theta if g is not None]

    # Apply gradients
    optim.apply_gradients(zip(grad_theta, model.trainable_variables + [alpha2]))

    return loss_ics_value, loss_bry_value, loss_res_value, loss, alpha2


# Plot L² Error
def plot_l2_error(tspace, relative_L2_error, filename="l2_error.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(tspace, relative_L2_error, marker='o', color='blue', label="L² Error")
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('Relative $L^2$ Error', fontsize=14)
    plt.title('Relative L² Error Over Time', fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# Plot Losses
def plot_losses(hist_ics, hist_bry, hist_res, hist, filename="losses.png"):
    plt.figure(figsize=(12, 6))
    plt.semilogy(hist, 'k-', label="Total Loss")
    plt.semilogy(hist_ics, 'b-', label="IC Loss")
    plt.semilogy(hist_bry, 'g-', label="BC Loss")
    plt.semilogy(hist_res, 'r-', label="Residual Loss")
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss Value (Log Scale)', fontsize=14)
    plt.title('Losses Over Epochs', fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# Plot 4 Computation Results
def plot_computation_results(xspace, exact_v_long, V, tspace, relative_L2_error, filename="computational_results.png"):
    Nt_grid = len(tspace)
    mid1 = Nt_grid // 3
    mid2 = 2 * Nt_grid // 3

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot 1: At t = 0
    axes[0].plot(xspace, exact_v_long[0], '--', color='blue', label='Exact')
    axes[0].plot(xspace, V.T[0], 'o', color='red', fillstyle='none', markersize=3, label='Predicted')
    axes[0].set_title(f'$t = {tspace[0]:.2f}$; $\\epsilon(\\ell^2)$ = {relative_L2_error[0]:.4f}')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$v(t, x)$')

    # Plot 2: Midpoint 1
    axes[1].plot(xspace, exact_v_long[mid1], '--', color='blue', label='Exact')
    axes[1].plot(xspace, V.T[mid1], 'o', color='red', fillstyle='none', markersize=3)
    axes[1].set_title(f'$t = {tspace[mid1]:.2f}$; $\\epsilon(\\ell^2)$ = {relative_L2_error[mid1]:.4f}')
    axes[1].set_xlabel('$x$')

    # Plot 3: Midpoint 2
    axes[2].plot(xspace, exact_v_long[mid2], '--', color='blue', label='Exact')
    axes[2].plot(xspace, V.T[mid2], 'o', color='red', fillstyle='none', markersize=3)
    axes[2].set_title(f'$t = {tspace[mid2]:.2f}$; $\\epsilon(\\ell^2)$ = {relative_L2_error[mid2]:.4f}')
    axes[2].set_xlabel('$x$')

    # Plot 4: Final time
    axes[3].plot(xspace, exact_v_long[-1], '--', color='blue', label='Exact')
    axes[3].plot(xspace, V.T[-1], 'o', color='red', fillstyle='none', markersize=3)
    axes[3].set_title(f'$t = {tspace[-1]:.2f}$; $\\epsilon(\\ell^2)$ = {relative_L2_error[-1]:.4f}')
    axes[3].set_xlabel('$x$')

    for ax in axes:
        ax.set_xticks(np.arange(-1, 4.1, 1.0))
    
    fig.tight_layout()
    
    # Save the plot to a file
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# Relative L² Error Calculation
def compute_relative_L2_error(exact_v_long, V, Nt_grid):
    relative_L2_error = []
    for i in range(Nt_grid):
        error = np.linalg.norm(exact_v_long[i] - V.T[i], 2) / np.linalg.norm(exact_v_long[i], 2)
        relative_L2_error.append(error)
    relative_error = np.average(relative_L2_error)
    return relative_L2_error, relative_error

# Main execution with training loop
if __name__ == "__main__":
    set_global_config(dtype='float64', seed=1234)
    pi, lb, ub = initialize_constants()
    
    # Load exact data
    exact_t, exact_x, exact_v, exact_t_long, exact_x_long, exact_v_long = load_exact_data()

    # Define number of data points
    N_0, N_b, N_r = 5000, 5000, 10000
    
    # Define bounds for boundary and initial conditions
    tmin, tmax, xmin, xmax = lb[0], ub[0], lb[1], ub[1]
    
    # Initial condition data
    t_0 = tf.ones((N_0,1), dtype=tf.float64) * tmin
    x_0 = tf.random.uniform((N_0,1), xmin, xmax, dtype=tf.float64)
    X_0 = tf.concat([t_0, x_0], axis=1)
    
    # Boundary condition data
    t_b = tf.random.uniform((N_b,1), tmin, tmax, dtype=tf.float64)
    x_b = xmin + (xmax - xmin) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=tf.float64)
    X_b = tf.concat([t_b, x_b], axis=1)
    
    # Collocation points (residual points)
    t_r = tf.random.uniform((N_r,1), tmin, tmax, dtype=tf.float64)
    x_r = tf.random.uniform((N_r,1), xmin, xmax, dtype=tf.float64)
    X_r = tf.concat([t_r, x_r], axis=1)
    
    ###########ININIAL COnditions############################
    # Initial and boundary condition values (v_0 and v_b)
  #  v_0 = tf.sin(pi * x_0)  # Example: sin(pi * x)
  #  v_b = tf.zeros_like(x_b)  # Example: boundary condition v=0
    
    v_0 = fun_v_0(x_0)
    v_b = fun_v_0(x_b)
    
    # Pack initial and boundary data into lists
    X_data = [X_0, X_b]
    v_data = [v_0, v_b]

    # Model definition
    model = init_model(lb, ub)

    # Training settings
    N_epoch = 10000
    optim = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Alpha2 variable
    alpha2 = tf.Variable(tf.random.uniform(shape=[], minval=1e-6, maxval=2.5e-3, dtype=tf.float64), 
                         constraint=lambda x: tf.clip_by_value(x, 1e-6, 2.5e-3), dtype=tf.float64, trainable=True)
    
    # History of loss values
    hist_ics, hist_bry, hist_res, hist = [], [], [], []
    hist_alpha2 = []

    # Training loop
    t0 = time()
    for i in range(N_epoch + 1):
        loss_ics_value, loss_bry_value, loss_res_value, loss, alpha2 = train_step(model, alpha2, X_r, X_data, v_data, optim)
        
        hist_ics.append(loss_ics_value.numpy())
        hist_bry.append(loss_bry_value.numpy())
        hist_res.append(loss_res_value.numpy())
        hist.append(loss.numpy())
        hist_alpha2.append(alpha2.numpy())

        # Print losses every 1000 epochs
        if i % 1000 == 0:
            print(f"Epoch {i:05d}: Loss = {loss:.8e}, Alpha2 = {alpha2:.8e}")

    print(f"Training completed in {time() - t0:.2f} seconds")
    
    # Compute the relative L² error after training
    Nt_grid = len(exact_t_long)  # Number of time steps
    Nx_grid = len(exact_x_long)  # Number of spatial points

    # Create a meshgrid for time and space
    T, X = np.meshgrid(exact_t_long, exact_x_long)

    # Flatten the meshgrid for input into the model
    T_flat = T.flatten()
    X_flat = X.flatten()

    # Stack time and space into a single input array of shape (N, 2)
    V = model(tf.cast(np.stack([T_flat, X_flat], axis=1), dtype=tf.float64))

    # Reshape the model output to the shape of the original grid (Nx_grid, Nt_grid)
    V = V.numpy().reshape(Nx_grid, Nt_grid)
    
    relative_L2_error, relative_error = compute_relative_L2_error(exact_v_long, V, Nt_grid)

    # 1. Plot L² Error
    plot_l2_error(exact_t_long, relative_L2_error, filename="./results/l2_error_plot.png")

    # 2. Plot Losses
    plot_losses(hist_ics, hist_bry, hist_res, hist, filename="./results/losses_plot.png")

    # 3. Plot Computation Results
    plot_computation_results(exact_x_long, exact_v_long, V, exact_t_long, relative_L2_error,  filename="./results/computation_results_plot.png")

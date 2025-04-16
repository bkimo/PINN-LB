#%% Notations
# u = the filtered velocity vector
#
#
# Built upon Dr. Maziar Raissi's PINNs - 
# https://github.com/maziarraissi/PINNs/tree/master/appendix/continuous_time_identification%20(Burgers)   
# 
#
#%% IMPORTING/SETTING UP PATHS
import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% LOCAL IMPORTS

#eqnPath = "../Burgers-alpha-MLP-PINN"
eqnPath = "../2.PINN-LB-TF2"
sys.path.append(eqnPath)
sys.path.append("utils")
sys.path.append("data")
from burgersutil import prep_data, plot_ide_cont_results
from neuralnetwork import NeuralNetwork
from logger import Logger
from plotting import newfig, savefig

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(5678)
tf.random.set_seed(5678)

#%% HYPER PARAMETERS
hp = {}
# Data size on the solution u
hp["N_u"] = 2000
# DeepNN topology (2-sized input [x t], L-hidden layers of n neurons, 1-sized output [u]
hp["layers"] = [2, 20, 20 , 20, 20, 20, 20, 20, 20, 1]

# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 0
hp["tf_lr"] = 0.001
hp["tf_b1"] = 0.9
hp["tf_eps"] = None   
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 50001
hp["nt_lr"] = 0.01 #Default = 1.0
hp["nt_ncorr"] = 50
hp["log_frequency"] = 1000


#%% DEFINING THE MODEL

class BurgersInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, ub, lb):
        super().__init__(hp, logger, ub, lb)

        # Defining the two additional trainable variables for identification
        self.lambda_1 = tf.Variable([0.0], dtype=self.dtype)
        self.lambda_2 = tf.Variable([-6.0], dtype=self.dtype)

    # The actual PINN
    def f_model(self, X_u):
        l1, l2 = self.get_params()
        # Separating the collocation coordinates
        x_f = tf.convert_to_tensor(X_u[:, 0:1], dtype=self.dtype)
        t_f = tf.convert_to_tensor(X_u[:, 1:2], dtype=self.dtype)

        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape1:
            # Watching the two inputs we’ll need later, x and t
            tape1.watch(x_f)
            tape1.watch(t_f)
            # Packing together the inputs
      #      X_f = tf.stack([x_f[:,0], t_f[:,0]], axis=1)


            # Getting the prediction
     #       u = self.model(X_f)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x_f)
                tape2.watch(t_f)
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch(x_f)
                    tape3.watch(t_f)
                    with tf.GradientTape(persistent=True) as tape4:
                        tape4.watch(x_f)
                        tape4.watch(t_f)
                        with tf.GradientTape(persistent=True) as tape5:
                            tape5.watch(x_f)
                            tape5.watch(t_f)
                            # Packing together the inputs
                            X_f = tf.stack([x_f[:,0], t_f[:,0]], axis=1)
                            # Getting the prediction
                            u = self.model(X_f)

                            # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
                            u_x = tape5.gradient(u, x_f)
                            u_t = tape5.gradient(u, t_f)
                        # Getting the other derivatives
                        u_xx = tape4.gradient(u_x, x_f)
                    u_tx = tape3.gradient(u_t, x_f)
            u_xxx = tape2.gradient(u_xx, x_f)
        u_txx = tape1.gradient(u_tx, x_f)

        # Letting the tape go
        del tape1, tape2, tape3, tape4, tape5

        # Buidling the PINNs
        return u_t + l1*u*u_x - l2*(u_txx + u*u_xxx)

    # Defining custom loss
    def loss(self, u, u_pred):
        f_pred = self.f_model(self.X_u)
        return tf.reduce_mean(tf.square(u - u_pred)) + \
               tf.reduce_mean(tf.square(f_pred))

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        var.extend([self.lambda_1, self.lambda_2])
        return var

    def get_weights(self):
        w = super().get_weights(convert_to_tensor=False)
        w.extend(self.lambda_1.numpy())
        w.extend(self.lambda_2.numpy())
        return tf.convert_to_tensor(w, dtype=self.dtype)

    def set_weights(self, w):
        super().set_weights(w)
        self.lambda_1.assign([w[-2]])
        self.lambda_2.assign([w[-1]])

    def get_params(self, numpy=False):
        l1 = self.lambda_1
        l2 = tf.exp(self.lambda_2)
        if numpy:
            return l1.numpy()[0], l2.numpy()[0]
        return l1, l2

    def fit(self, X_u, u):
        self.X_u =  tf.convert_to_tensor(X_u, dtype=self.dtype)
        super().fit(X_u, u)

    def predict(self, X_star):
        u_star = self.model(X_star)
        f_star = self.f_model(X_star)
        return u_star.numpy(), f_star.numpy()

#%% TRAINING THE MODEL

# Getting the data
path = os.path.join(eqnPath, "data")
x, t, X, T, Exact_u, X_star, u_star, \
        X_u_train, u_train, ub, lb = prep_data(path, hp["N_u"], noise=0.0)
lambdas_star = (1.0, -0.7)

# Creating the model
logger = Logger(hp)
pinn = BurgersInformedNN(hp, logger, ub, lb)

pinn.summary()

# Defining the error function and training
def error():
    u_pred, _ = pinn.predict(X_star)
    return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
#def error():
#    l1, l2 = pinn.get_params(numpy=True)
#    l1_star, l2_star = lambdas_star
#    error_lambda_1 = np.abs(l1 - l1_star) / l1_star 
#    error_lambda_2 = np.abs(l2 - l2_star) / l2_star
#    return (error_lambda_1 + error_lambda_2) / 2
###    return error_lambda_1
logger.set_error_fn(error)
pinn.fit(X_u_train, u_train) # No compile before fit? -> optimizing = compile (?)

#???? Getting the model predictions (validation(?)), from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)
lambda_1_pred, lambda_2_pred = pinn.get_params(numpy=True)

error_lambda1 = np.abs(lambda_1_pred-lambdas_star[0])
print("l1: ", lambda_1_pred)
print("Abs Error of l1:", error_lambda1)
print("l2: ", lambda_2_pred)


######################################################################
############################# PLOTTING ###############################
######################################################################    

fig, ax = newfig(1.0, 1.4)
ax.axis('off')

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])
    
h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
    
ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)
    
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[-1]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
ax.set_title('$u(t,x)$', fontsize = 10)
    
####### Row 1: u(t,x) slices ##################   
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.3)
    
ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact_u[25,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = %.3f$' % t[25], fontsize = 10)
#ax.axis('square')
#ax.grid()
ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
ax.set_xlim([-2.1,4.1])
ax.set_ylim([-0.1,1.1])
    
ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact_u[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
#ax.grid()
ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
ax.set_xlim([-2.1,4.1])
ax.set_ylim([-0.1,1.1])
ax.set_title('$t = %.3f$' % t[50], fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact_u[-1, :], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[-1,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')

ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
ax.set_xlim([-2.1,4.1])
ax.set_ylim([-0.1,1.1])  
ax.set_title('$t = %.3f$' % t[-1], fontsize = 10)
    
####### Row 2: Identified PDE ##################    
gs2 = gridspec.GridSpec(1, 3)
gs2.update(top=1.0-2.0/3.0-0.1, bottom=1.0-2.5/3.0-0.1, left=0.1, right=0.9, wspace=0.0)
    
ax = plt.subplot(gs2[:, :])
ax.axis('off')
s1 = r'$\begin{tabular}{ |c|c| }  \hline Data PDE & $u_t + u u_x  = 0$ \\  \hline Identified PDE & '
s2 = r'$u_t + %.5f u u_x = %.7f (u_{txx} + u  u_{xxx})$ \\  \hline ' % (lambda_1_pred, lambda_2_pred)
s3 = r'Error in $\lambda_1$ & '
s4 = r'%.5f  \\  \hline ' % (error_lambda1)
s5 = r'L2-Error & '
s6 = r'%.5f  \\  \hline ' % (error_lambda1)
s5 = r'\end{tabular}$'
s = s1+s2+s3+s4+s5
ax.text(0.1,0.1,s)

#savefig('./results/ep11-2-9-1')
savefig('temp1')
plt.show()

"""
@author: Archie Huang
Built upon Dr. Maziar Raissi's PINNs - https://github.com/maziarraissi/PINNs

Use Tensorflow 1.15
"""

import random
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
N_EPOCH= 300000
ALPHA2 = 0.02**2

se = 4
np.random.seed(se)
tf.set_random_seed(se)

# PINN Class for LWT-alpha
class PhysicsInformedNNalpha:
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u = u

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': N_EPOCH,
                                                                           'maxfun': N_EPOCH,
                                                                           'maxcor': 50,
                                                                           'maxls': 20,
                                                                           'ftol' : 2.220446049250313e-09})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u

    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        # params - v_f = 25m/s, rho_m = 0.15 veh/m
        f = 0.15 * u_x - 0.012 * u * u_x - 0.006 * u_t +  ALPHA2*u_x*u_xx # alpha = 0.025
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
        return u_star, f_star

class PhysicsInformedNNepsilon:
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u = u

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': N_EPOCH,
                                                                           'maxfun': N_EPOCH,
                                                                           'maxcor': 50,
                                                                           'maxls': 20,
                                                                           'ftol' : 2.220446049250313e-09})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u

    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        # params - v_f = 25m/s, rho_m = 0.15 veh/m
        f = 0.15 * u_x - 0.012 * u * u_x - 0.006 * u_t - ALPHA2 * u_xx ## physics-cost with diffusion term
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
        return u_star, f_star
    
# PINN Class for Exact LWT
class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u = u

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': N_EPOCH,
                                                                           'maxfun': N_EPOCH,
                                                                           'maxcor': 50,
                                                                           'maxls': 20,
                                                                           'ftol' : 2.220446049250313e-09})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u

    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        # params - v_f = 25m/s, rho_m = 0.15 veh/m
        #f = 0.15 * u_x - 0.012 * u * u_x - 0.006 * u_t 
        f = 0.15 * u_x - 0.012 * u * u_x - 0.006 * u_t - 0.000625 * u_xx ## physics-cost with diffusion term
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
        return u_star, f_star



if __name__ == "__main__":

    N_u = 10000
    N_f = 30000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    #layers = [2, 20, 30, 30, 30, 30, 30, 30, 30, 1]
    data = scipy.io.loadmat('data/synthetic.mat')
    t = data['tScale'].T.flatten()[:,None]
    x = data['xScale'].T.flatten()[:,None]
    Exact = np.real(data['v'])
    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    lb = X_star.min(0)
    ub = X_star.max(0)

    ################################# CAV Plot ####################################
    N_cav = 500 # Max = 508
    cav = np.real(data['N'])
    idx_cav = np.sort(random.sample(range(int(cav.min()), int(cav.max())), N_cav))
    contr = plt.contour(X, T, cav, idx_cav)
    plt.title('CAV Trajectories', fontsize = 16)
    plt.ylabel('Time $t$ (s)', fontsize = 14)
    plt.xlabel('Location $x$ (m)', fontsize=14)
    plt.savefig('results/cav{}_trajectory.pdf'.format(N_cav))
  #  plt.savefig('results/cav{}_trajectory.eps'.format(N_cav))
    #plt.show()
    ################################# CAV Plot ####################################

    ################################# CAV Samples #################################
    X_u_train = contr.collections[0].get_paths()[0].vertices
    for a in range(1, N_cav):
        path = contr.collections[a].get_paths()[0]
        ver = path.vertices
        X_u_train = np.vstack([X_u_train, ver])
    TDP = len(X_u_train)
    print("Total Trajectory Data Points: ", len(X_u_train), '\n')
    ################################# CAV Samples #################################

    ################################# Filters #####################################
    # RSU communication range and locations
    comm_range = 300
    loca = [0, 1000, 2000, 3000, 4000, 5000]

    mask = ((X_u_train[:, 0] < loca[0] + comm_range) |
            ((loca[1] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[1] + comm_range)) |
            ((loca[2] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[2] + comm_range)) |
            ((loca[3] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[3] + comm_range)) |
            ((loca[4] - comm_range < X_u_train[:, 0]) & (X_u_train[:, 0] < loca[4] + comm_range)) |
            (loca[5] - comm_range < X_u_train[:, 0]))
    X_u_train = X_u_train[mask]

    X_u_train = np.rint(X_u_train)
    X_u_train[:, 0] = np.rint(X_u_train[:, 0] / 5) * 5  # fit for 5-meter bins
    X_u_train = np.unique(X_u_train, axis=0).astype(int)
    RSU = len(X_u_train)
    print("Total RSU Data Points: ", len(X_u_train), '\n')
    ################################# Filters #####################################

    ############################### Training Data #################################
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.empty([N_u, 1])
    for i in range(0, N_u):
        u_train[i] = Exact[X_u_train[i][1]][int(X_u_train[i][0] / 5)]
    ############################### Training Data #################################

    # PINN Model for LWR-alpha
    model = PhysicsInformedNNalpha(X_u_train, u_train, X_f_train, layers, lb, ub)
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)
    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % error_u)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)

    
    # PINN Model for LWR
    model2 = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
    start_time2 = time.time()
    model2.train()
    elapsed2 = time.time() - start_time2
    print('Training time: %.4f' % elapsed2)
    u_pred2, f_pred2 = model2.predict(X_star)
    error_u2 = np.linalg.norm(u_star-u_pred2,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % error_u2)
    U_pred2 = griddata(X_star, u_pred2.flatten(), (X, T), method='cubic')
    Error2 = np.abs(Exact - U_pred2)

    # PINN Model for LWR-epsilone
    model3 = PhysicsInformedNNepsilon(X_u_train, u_train, X_f_train, layers, lb, ub)
    start_time3 = time.time()
    model3.train()
    elapsed3 = time.time() - start_time3
    print('Training time: %.4f' % elapsed3)
    u_pred3, f_pred3 = model3.predict(X_star)
    error_u3 = np.linalg.norm(u_star-u_pred3,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % error_u3)
    U_pred3 = griddata(X_star, u_pred3.flatten(), (X, T), method='cubic')
    Error3 = np.abs(Exact - U_pred3)

    #### Print Results
    #### Print Results
    print(' ')
    print("Total Trajectory Data Points: ", TDP, '\n')
    print("Total RSU Data Points: ", len(X_u_train), '\n')
    
    # Calculate MSE for each model
    mse_lwt = np.mean((u_star - u_pred2.flatten()[:,None])**2)
    mse_lwt_alpha = np.mean((u_star - u_pred.flatten()[:,None])**2)
    mse_lwt_epsilon = np.mean((u_star - u_pred3.flatten()[:,None])**2)
    
    print('Post Logs')
    print(' ')
    print('LWT Relative Error u: %e' % error_u2)
    print('LWT Abs Error u: %e' %  np.mean(Error2))
    print('LWT MSE: %e' % mse_lwt)
    print('LWT Training time: %.4f' % elapsed2)
    print(' ')
    print('LWT-alpha Relative Error u: %e' % error_u)
    print('LWT-alpha Abs Error u: %e' %  np.mean(Error))
    print('LWT-alpha MSE: %e' % mse_lwt_alpha)
    print('LWT-alpha Training time: %.4f' % elapsed)
    print(' ')
    print('LWT-epsilon Relative Error u: %e' % error_u3)
    print('LWT-epsilon Abs Error u: %e' %  np.mean(Error3))
    print('LWT-epsilon MSE: %e' % mse_lwt_epsilon)
    print('LWT-epsilon Training ticpme: %.4f' % elapsed3)
    print(' ')
    
# Plot using the best optimizer's results
    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
        ]
    }
    mpl.rcParams.update(pgf_with_latex)
    fig = plt.figure(figsize=(15, 12))

    # Create 2x2 subplots
    gs = gridspec.GridSpec(2, 2)
    gs.update(top=0.95, bottom=0.1, left=0.1, right=0.9, wspace=0.3, hspace=0.3)

    # Reference Solution (1,1)
    ax1 = plt.subplot(gs[0, 0])
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.yaxis.set_major_locator(MultipleLocator(60))
    h1 = ax1.imshow(Exact, interpolation='nearest', cmap='RdBu',
                    extent=[x.min(), x.max(), t.min(), t.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cax1.tick_params(labelsize=20)
    fig.colorbar(h1, cax=cax1, ticks=[5, 10, 15, 20])
    #ax1.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax1.set_ylabel('Time $t$ (s)', fontsize=20)
    ax1.set_xlabel('Location $x$ (m)', fontsize=20)
    ax1.set_title('Reference Solution', fontsize=18)


    # LWR-alpha (1,2)
    ax2 = plt.subplot(gs[0, 1])
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.yaxis.set_major_locator(MultipleLocator(60))
    h2 = ax2.imshow(U_pred, interpolation='nearest', cmap='RdBu',
                    extent=[x.min(), x.max(), t.min(), t.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cax2.tick_params(labelsize=20)
    fig.colorbar(h2, cax=cax2, ticks=[5, 10, 15, 20])
    ax2.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax2.set_ylabel('Time $t$ (s)', fontsize=20)
    ax2.set_xlabel('Location $x$ (m)', fontsize=20)
    ax2.set_title('LWR-$\\alpha$ ($\\alpha = {:.6f}$)'.format(np.sqrt(ALPHA2)), fontsize=18)

    # LWR (2,1)
    ax3 = plt.subplot(gs[1, 0])
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.yaxis.set_major_locator(MultipleLocator(60))
    h3 = ax3.imshow(U_pred2, interpolation='nearest', cmap='RdBu',
                    extent=[x.min(), x.max(), t.min(), t.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    cax3.tick_params(labelsize=20)
    fig.colorbar(h3, cax=cax3, ticks=[5, 10, 15, 20])
    ax3.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax3.set_ylabel('Time $t$ (s)', fontsize=20)
    ax3.set_xlabel('Location $x$ (m)', fontsize=20)
    ax3.set_title('LWR', fontsize=18)

    # LWR-epsilon (2,2)
    ax4 = plt.subplot(gs[1, 1])
    ax4.tick_params(axis='both', which='major', labelsize=20)
    ax4.yaxis.set_major_locator(MultipleLocator(60))
    h4 = ax4.imshow(U_pred3, interpolation='nearest', cmap='RdBu',
                    extent=[x.min(), x.max(), t.min(), t.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    cax4.tick_params(labelsize=20)
    fig.colorbar(h4, cax=cax4, ticks=[5, 10, 15, 20])
    ax4.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=0.8, clip_on=False)
    ax4.set_ylabel('Time $t$ (s)', fontsize=20)
    ax4.set_xlabel('Location $x$ (m)', fontsize=20)
    ax4.set_title('LWR-$\\epsilon$ ($\\epsilon = {:.6f}$)'.format(ALPHA2), fontsize=18)

    plt.savefig('results/cav{}_comparison_alpha0.025_epsilon0.000625.pdf'.format(N_cav))
    plt.show()
    print("Total RSU Data Points: ", len(X_u_train), '\n')
    ################################# Plot #################################
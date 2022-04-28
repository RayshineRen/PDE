import sys
sys.path.insert(0, '../Storage/')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time
from storage_utils import dumpTotalLoss
from log_utils import logTime, logRelativeError
from plot_utils import plotting
from file_utils import arrangeFiles
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

np.random.seed(1234)
tf.set_random_seed(1234)

class gPINN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, W_x, W_t, LR):
        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]
        self.u = u
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.layers = layers
        self.nu = nu
        self.w_x  = W_x
        self.w_t  = W_t
        self.loss_log = []
        self.loss_r_log = []
        self.loss_b_log = []
        self.loss_g_log = []
        self.Nf = self.Nf = X_f.shape[0] - X_u.shape[0]

        self.weights, self.biases = self.initilize_NN(layers)
        self.sess = tf.Session()

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf   = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss_b = tf.reduce_mean(tf.square(self.u_pred - self.u_tf))
        self.loss_r = tf.reduce_mean(tf.square(self.f_pred))
        self.loss_g = self.gradient_enhance()
        self.loss   = self.loss_b + self.loss_r + self.loss_g

        self.LR = LR
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initilize_NN(self, layers):
        weights = []
        biases  = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
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
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - self.nu * u_xx
        return f

    def gradient_enhance(self):
        g_x = tf.reduce_mean(tf.square(tf.gradients(self.f_pred, self.x_f_tf)[0]))
        g_t = tf.reduce_mean(tf.square(tf.gradients(self.f_pred, self.t_f_tf)[0]))
        return self.w_x * g_x + self.w_t * g_t

    def train(self, nIter=40000, tresh=1e-32):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        start_time = time.time()
        loss_value = 0
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value  = self.sess.run(self.loss, tf_dict)
            loss_valueb = self.sess.run(self.loss_b, tf_dict)
            loss_valueg = self.sess.run(self.loss_g, tf_dict)
            loss_valuer = self.sess.run(self.loss_r, tf_dict)
            self.loss_log.append(loss_value)
            self.loss_b_log.append(loss_valueb)
            self.loss_r_log.append(loss_valuer)
            self.loss_g_log.append(loss_valueg)
            if loss_value < tresh:
                print('It: %d, Loss: %.3e' % (it, loss_value))
                break
            if it % 2000 == 0:
                elapsed = time.time() - start_time
                str_print = 'It: %d, Lossb: %.3e, Lossg: %.3e, Lossr: %.3e, Time: %.2f'
                print(str_print % (it, loss_valueb, loss_valueg, loss_valuer, elapsed))
        end_time = time.time()
        print("training time %f, loss %f"%(end_time - start_time, loss_value))
        self.training_time = end_time - start_time

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf:X_star[:,1:2]})
        return u_star

nu = 0.01 / np.pi
iter = 20000
N_u = 100
N_f = 2000
LR = 0.001
w_x = 0.01
w_t = 0.01
layers = [2] + [40] * 5 + [1]
path = r'D:\Documents\grade4term1\PDE\数学基础\NN\TF_learn'
data = scipy.io.loadmat(path + '/Burgers/burgers_shock.mat')

x = data['x'].flatten()[:, None]
t = data['t'].flatten()[:, None]
Exact =np.real(data['usol']).T
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]
# BC / IC
# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)
# 初始点
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
# x=-1的边界点
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]
# x=1的边界点
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]
X_u_train = np.vstack([xx1, xx2, xx3]) #X_u_train.shape=(456, 2)
u_train = np.vstack([uu1, uu2, uu3])
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False) # 抽取N_u个点
X_u_train = X_u_train[idx]
u_train = u_train[idx]

def train_f(N_f, iter):
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    model = gPINN(X_u_train, u_train, X_f_train,
                layers, lb, ub, nu, w_x, w_t, LR)
    model.train(nIter=iter)
    u_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
    logTime(model)
    logRelativeError(model, error_u)
    u_pred = u_pred.reshape(-1, 256)
    plotting(X, T, Exact, u_pred)
    dumpTotalLoss(model)
    arrangeFiles(model, iter)

Nfs = [500, 1000, 2500, 5000, 10000]
for N_f in Nfs:
    train_f(N_f, iter)
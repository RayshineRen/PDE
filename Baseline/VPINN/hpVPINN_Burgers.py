import sys
sys.path.insert(0, '../Storage/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
from storage_utils import dumpTotalLoss
from log_utils import logTime, logRelativeError
from plot_utils import plotting
from file_utils import arrangeFiles
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights, GaussJacobiWeights
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class VPINN:
    def __init__(self, X_u, u, X_r, X_quad, W_x_quad, \
                 Y_quad, W_y_quad, F_exact_total, grid_x, grid_y, layers, \
                 LR, lossb_weight, lossv_weight, activation):
        self.x         = X_u[:, 0:1]
        self.y         = X_u[:, 1:2]
        self.u         = u
        self.x_r       = X_r[:, 0:1]
        self.y_r       = X_r[:, 1:2]
        self.xquad     = X_quad    # 不需要训练点(xf, f) 只需要求积点和边界点
        self.yquad     = Y_quad
        self.wquad_x   = W_x_quad
        self.wquad_y   = W_y_quad
        self.F_ext_total = F_exact_total
        self.grid_x    = grid_x
        self.grid_y    = grid_y
        self.activation= activation
        self.NEx       = grid_x.shape[0] - 1
        self.NEy       = grid_y.shape[0] - 1
        self.loss_log = []
        self.loss_r_log = []
        self.loss_b_log = []
        self.loss_v_log = []
        self.Nf = self.Nf = X_r.shape[0] - X_u.shape[0]

        self.x_tf   = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.y_tf   = tf.placeholder(tf.float64, shape=[None, self.y.shape[1]])
        self.u_tf   = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])
        self.x_r_tf   = tf.placeholder(tf.float64, shape=[None, self.x_r.shape[1]])
        self.y_r_tf   = tf.placeholder(tf.float64, shape=[None, self.y_r.shape[1]])

        self.weights, self.biases = self.initialize_NN(layers)
        self.u_NN_pred = self.net_u(self.x_tf, self.y_tf)   # 边界点预测值

        self.lossb  = tf.reduce_mean(tf.square(self.u_tf - self.u_NN_pred))
        self.lossv  = self.variational_loss()
        self.lossr  = self.residual_loss(self.x_r_tf, self.y_r_tf)
        self.loss   = lossb_weight * self.lossb + lossv_weight * self.lossv + self.lossr

        self.LR = LR
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim), dtype=np.float64)
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64), dtype=tf.float64)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = self.activation(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y):
        u = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        return u

    def residual_loss(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - 0.01 / np.pi * u_xx
        return tf.reduce_mean(tf.square(f))

    def net_du(self, x, y):
        u    = self.net_u(x, y)
        d1ux = tf.gradients(u, x)[0]
        d1uy = tf.gradients(u, y)[0]
        d2ux = tf.gradients(d1ux, x)[0]
        return d1ux, d2ux, d1uy

    # 构造测试函数集
    def Test_fcn(self, N_test, x):
        test_total = []
        for n in range(1, N_test+1):
            test = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
            test_total.append(test)
        return np.asarray(test_total)

    # variational loss
    def variational_loss(self):
        varloss_total = 0
        for e_y in range(self.NEy):
            for e_x in range(self.NEx):
                F_ext_element  = self.F_ext_total[e_y*self.NEy + e_x]     # 定位子区域，此形式中F_ext恒0
                Ntest_element  = int(np.sqrt(np.shape(F_ext_element)[0])) # 子区域的测试函数个数 x=y
                x_quad_element = tf.constant(self.grid_x[e_x] + \
                                 (self.grid_x[e_x + 1] - self.grid_x[e_x])\
                                             / 2*(self.xquad+1))          # 将求积点映射到子区域区间内
                jacobian_x     = (self.grid_x[e_x + 1] - self.grid_x[e_x]) / 2  # 系数
                # 测试函数及其微分 global(用xquad计算)
                testx_quad_element = self.Test_fcn(Ntest_element, self.xquad)

                y_quad_element = tf.constant(self.grid_y[e_y] + \
                                 (self.grid_y[e_y + 1] - self.grid_y[e_y]) \
                                             / 2*(self.yquad+1))
                jacobian_y     = (self.grid_y[e_y + 1] - self.grid_y[e_y]) / 2
                # 测试函数及其微分
                testy_quad_element = self.Test_fcn(Ntest_element, self.yquad)
                # PDE及其微分
                u_NN_quad_element = self.net_u(x_quad_element, y_quad_element)
                d1ux_NN_quad_element, d2ux_NN_quad_element, \
                d1uy_NN_quad_element = self.net_du(x_quad_element, y_quad_element)

                U_NN_element = []
                for phi_y in testy_quad_element:        # 对y积分
                    for phi_x in testx_quad_element:    # 对x积分
                        inte1_x = jacobian_x*tf.reduce_sum(self.wquad_x*\
                            (d1uy_NN_quad_element + u_NN_quad_element*d1ux_NN_quad_element -\
                             0.01/np.pi*d2ux_NN_quad_element)*phi_x)  # 权函数 * PDE * 测试函数
                        inte2_x = jacobian_y*tf.reduce_sum(self.wquad_y*inte1_x*phi_y)
                        U_NN_element.append(inte2_x)
                U_NN_element = tf.reshape(U_NN_element, (-1, 1))
                Res_NN_element = U_NN_element - F_ext_element
                loss_element   = tf.reduce_mean(tf.square(Res_NN_element))
                varloss_total += loss_element
        return varloss_total

    def predict(self, x, y):
        u_pred  = self.sess.run(self.u_NN_pred, {self.x_tf: x, self.y_tf: y})
        return u_pred

    def train(self, nIter=40000, tresh=1e-32):
        tf_dict = {self.x_tf: self.x, self.y_tf : self.y, self.u_tf: self.u, \
                   self.x_r_tf: self.x_r, self.y_r_tf: self.y_r}
        start_time = time.time()
        loss_value = 0
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value  = self.sess.run(self.loss, tf_dict)
            loss_valueb = self.sess.run(self.lossb, tf_dict)
            loss_valuev = self.sess.run(self.lossv, tf_dict)
            loss_valuer = self.sess.run(self.lossr, tf_dict)
            self.loss_log.append(loss_value)
            self.loss_b_log.append(loss_valueb)
            self.loss_r_log.append(loss_valuer)
            self.loss_v_log.append(loss_valuev)
            if loss_value < tresh:
                print('It: %d, Loss: %.3e' % (it, loss_value))
                break
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                str_print = 'It: %d, Lossb: %.3e, Lossv: %.3e, Lossr: %.3e, Time: %.2f'
                print(str_print % (it, loss_valueb, loss_valuev, loss_valuer, elapsed))
        end_time = time.time()
        print("training time %f, loss %f"%(end_time - start_time, loss_value))
        self.training_time = end_time - start_time

LR = 0.001
iter = 20000
Opt_Niter = 40000
Opt_tresh = 2e-32
NEx = 1
NEy = 1
Net_layer  = [2] + [40] * 5 + [1]
activation = tf.tanh
Nx_testfcn = 5
Ny_testfcn = 5
Nx_Quad = 50
Ny_Quad = 50
lossb_weight = 1
lossv_weight = 1
N_u = 100
N_f = 2000

#++++++++++++++++++++++++++++
path = r'D:\Documents\grade4term1\PDE\数学基础\NN\TF_learn'
data = scipy.io.loadmat(path + '/Burgers/burgers_shock.mat')
x = data['x']
t = data['t']
Exact =np.real(data['usol']).T
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]
#++++++++++++++++++++++++++++
# IC / BC points
# Doman bounds
lb = x_l, y_l = X_star.min(0)
ub = x_r, y_h = X_star.max(0)
# 初始点
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
# x=-1的边界点
xx2 = np.hstack((X[:,0:1], T[:, 0:1]))
uu2 = Exact[:,0:1]
# x=1的边界点
xx3 = np.hstack((X[:,-1:], T[:, -1:]))
uu3 = Exact[:,-1:]
X_u_train = np.vstack([xx1, xx2, xx3])                         #X_u_train.shape=(456, 2)
u_train = np.vstack([uu1, uu2, uu3])
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False) # 抽取N_u个点
X_u_train = X_u_train[idx]
u_train = u_train[idx]

# 区域划分
delta_x = (x_r - x_l) / NEx
# 区域网格点 eg. 均分为两个区域[-1, 0, 1]
grid_x  = np.asarray([x_l + i*delta_x for i in range(NEx+1)])
# 每个区域内测试函数的数目，目前都一样，结果只取决于区域内的求积点
Nx_testfcn_total = np.array((len(grid_x) - 1)*[Nx_testfcn])
delta_y = (y_h - y_l) / NEy
grid_y  = np.asarray([y_l + i*delta_y for i in range(NEy+1)])
Ny_testfcn_total = np.array((len(grid_y) - 1)*[Ny_testfcn])
# 计算Fk NEx * NEy 个区域
F_ext = np.array((Nx_testfcn * Ny_testfcn) * [0])[:, None]
F_ext_total = np.tile(F_ext, (NEx * NEy, 1)).reshape(-1, Nx_testfcn * Ny_testfcn, 1)

#++++++++++++++++++++++++++++
# 测试函数迭代 高斯雅各比迭代 测试函数彼此正交 最终总共构造N_testfcn个测试函数
def Test_fcn(n, x):
   test  = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
   return test

[x_quad, w_quad_x] = GaussLobattoJacobiWeights(Nx_Quad, 0, 0)
[y_quad, w_quad_y] = GaussLobattoJacobiWeights(Ny_Quad, 0, 0)
#++++++++++++++++++++++++++++
# Quadrature points
X_quad   = x_quad[:, None]
W_quad_x = w_quad_x[:, None]
Y_quad   = y_quad[:, None]
W_quad_y = w_quad_y[:, None]
#++++++++++++++++++++++++++++

def train_f(N_f, iter):
    X_f_train = lb + (ub - lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    model = VPINN(X_u_train, u_train, X_f_train, X_quad, W_quad_x, \
              Y_quad, W_quad_y, F_ext_total, grid_x, grid_y, Net_layer, \
              LR, lossb_weight, lossv_weight, activation)
    model.train(nIter = iter)
    u_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star-u_pred, 2) / np.linalg.norm(u_star, 2)
    logTime(model)
    logRelativeError(model, error_u)
    u_pred = u_pred.reshape(-1, 256)
    plotting(X, T, Exact, u_pred)
    dumpTotalLoss(model)
    arrangeFiles(model, iter)

Nfs = [500, 1000, 2500, 5000, 10000]
for N_f in Nfs:
    train_f(N_f, iter)

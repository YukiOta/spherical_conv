# coding: utf-8

import numpy as np
from utilities import im2col, col2im
from utilities import fish2col, col2fish, pool_backward
from keras import backend as K
import tensorflow as tf

"""
By using CuPy, try to adapt this program
to GPU computations.
"""


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def binary_cross_entropy_error(y, t):
    y = tf.convert_to_tensor(y, np.float32)
    t = tf.convert_to_tensor(t, np.float32)
    binary_error = K.mean(K.binary_crossentropy(t, y), axis=-1)
    binary_error = tf.Session().run(binary_error)
    return binary_error


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        try:
            t = t.argmax(axis=1)
        except:
            t = t.argmax(axis=0)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)


"""layers (common/layers.py)
ネットワークを組み立てるのに必要な層(layer)クラスの定義
 - Relu
 - sigmoid
 - affine
 - convolution
 - pooling
"""


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        self.loss = binary_cross_entropy_error(self.t, self.y)
        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None,
                 running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, A, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean \
                + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        # self.gamma = self.gamma.reshape(-1)
        # self.beta = self.beta.reshape(-1)
        # out = self.gamma * xn + self.beta
        out = xn * self.gamma + self.beta

        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, A, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        # print("SoftmaxWithLoss Forward")
        self.t = t
        # self.y = softmax(x)
        self.y = sigmoid(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        # print("SoftmaxWithLoss Backward")
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class Convolution:  # {{{
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # max Pooling
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,
                    self.pool_w, self.stride, self.pad)

        return dx  # }}}


class Affine_FISH:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self, x):
        # print("Affine Forward")
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        # print("Affine Backward")
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx


class Convolution_FISH:
    def __init__(self, W, b, level, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.level = level

        # 中間データ
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        '''
        N: データ数
        C: チャンネル数
        A: area数 (=5)
        I: i座標
        J: j座標
        '''
        # print("Convlution Forward")
        N, C, A, I, J = x.shape
        # print("Input x shape: " + str(x.shape))
        # print("Input W shape: " + str(self.W.shape))
        FN, C, cell_num = self.W.shape

        col = fish2col(x, level=self.level)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = col2fish(data_num=N, input_mat=out, level=self.level,
                       FN=FN, cell_num=cell_num)
        # print("out shape: " + str(out.shape))

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        '''
        dout.shape -> (N, C, A, I, J)
        '''
        # print("Convolution Backward")
        FN, FC, cell_num = self.W.shape
        # print("Input W shape: " + str(self.W.shape))
        N = dout.shape[0]
        dout = self.dout2dout(dout)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, FC, cell_num)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2fish(
            data_num=N, input_mat=dcol, level=self.level, FN=FC,
            backward=True)
        return dx

    def dout2dout(self, dout):
        N, C, A, I, J = dout.shape
        for data in range(N):
            top = dout[data, :, 1, 0, 1]
            top = top.reshape(-1, C)
            bottom = dout[data, :, 1, I - 1, J - 1]
            bottom = bottom.reshape(-1, C)
            out_tmp = dout[data, :,  1:6, 1:I - 1, 1:J - 1]
            out_tmp = out_tmp.reshape(-1, C)
            # print(out_tmp.shape)
            # print(top.shape)
            # print(bottom.shape)
            if data == 0:
                array_out = np.concatenate((top, out_tmp, bottom), axis=0)
            else:
                array_tmp = np.concatenate((top, out_tmp, bottom), axis=0)
                array_out = np.concatenate((array_out, array_tmp), axis=0)
        return array_out


class Pooling_FISH:
    def __init__(self, cell_num, level, stride=1, pad=0):
        # self.pool_h = pool_h
        # self.pool_w = pool_w
        self.cell_num = cell_num
        self.stride = stride
        self.pad = pad
        self.level = level
        self.Q = 2**level

        self.x = None
        self.arg_max = None

    def forward(self, x):
        '''
        N: データ数
        C: チャンネル数
        A: area数 (=5)
        I: i座標
        J: j座標
        '''
        # print("Pooling Forward")
        N, C, A, I, J = x.shape
        # print("Input x shape: " + str(x.shape))
        col = fish2col(x, level=self.level)
        # フィルターごとに、近傍の中で一番大きい値を持ってくる
        col = col.reshape(-1, self.cell_num)

        # mac pooling
        # 一番大きい値をもつ位置のインでくっすを取得
        arg_max = np.argmax(col, axis=1)
        # max pooling
        out = np.max(col, axis=1)
        out = col2fish(data_num=N, input_mat=out, level=self.level,
                       FN=C, cell_num=self.cell_num, pool=True)
        # print(out.shape)
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        '''
        dout.shape -> (N, C, A, I, J)
        '''
        # print("Pooling Backward")
        _N, _C, _A, _I, _J = dout.shape
        # print("dout.shape"+str(dout.shape))
        dout_level = self.level - 1
        _dmax = np.zeros((_N, _C, _A, self.Q + 2, 2 * self.Q + 2))
        col_dmax = fish2col(input_data=_dmax, level=self.level)
        col_dout = fish2col(input_data=dout, level=dout_level)
        value = pool_backward(col_dout=col_dout, dmax=_dmax, level=dout_level)

        col_dmax = col_dmax.reshape((-1, self.cell_num), order="F")
        col_dmax[np.arange(self.arg_max.size), self.arg_max] = value
        col_dmax = col_dmax.reshape(-1, col_dmax.shape[-1]*_C)
        dx = col2fish(data_num=_N, input_mat=col_dmax, level=self.level, FN=_C)
        return dx

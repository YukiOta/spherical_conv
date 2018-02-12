# coding: utf-8

import sys
import os
import pickle
import numpy as np
from collections import OrderedDict
from layers_and_functions import *
from tqdm import tqdm
import time
from keras import backend as K


class SphereConv:
    '''球面コンボリューションを行う
    とりあえずの構造は，
    INPUT - CONV - POOL - AFFINE - RELU - AFFINE - SOFTMAX
    で頑張る

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）(今回の場合は)
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）(今回の場合は257個のクラス分類)
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定

    '''

    def __init__(self, input_dim=(3, 6, 18, 34),
                 conv_param={'filter_num': 32,
                             'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std='he',
                 level=4):
        self.level = level
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        pool_output_size = filter_num * 6 * \
            (2**(level - 1) + 2) * (2**level + 2)
        conv_out = filter_num * 6 * (2**level + 2) * \
            (2**(level + 1) + 2)

        if weight_init_std == 'he':
            weight_init_std = np.sqrt(2.0 / np.prod(input_dim))
        # conv_out = int(filter_num * np.prod(input_dim[1:]))
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size)
        self.params['b1'] = np.zeros(filter_num)

        self.params['W2'] = weight_init_std * \
            np.random.randn(filter_num, filter_num, filter_size)
        self.params['b2'] = np.zeros(filter_num)

        self.params['W3'] = weight_init_std * \
            np.random.randn(conv_out, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        middle_shape = 32 * 6 * 18 * 34
        self.params['gamma1'] = np.ones(middle_shape)
        self.params['beta1'] = np.zeros(middle_shape)
        self.params['gamma2'] = np.ones(middle_shape)
        self.params['beta2'] = np.zeros(middle_shape)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution_FISH(W=self.params['W1'], b=self.params['b1'], level=self.level,
                                                stride=conv_param['stride'], pad=conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Batch1'] = BatchNormalization(gamma=self.params['gamma1'], beta=self.params['beta1'])

        self.layers['Conv2'] = Convolution_FISH(W=self.params['W2'], b=self.params['b2'], level=self.level,
                                                stride=conv_param['stride'], pad=conv_param['pad'])
        self.layers['Batch2'] = BatchNormalization(gamma=self.params['gamma2'], beta=self.params['beta2'])
        self.layers['Relu2'] = Relu()

        self.layers['Affine1'] = Affine_FISH(
            self.params['W3'], self.params['b3'])
        self.layers['Relu2'] = Relu()

        self.layers['Affine2'] = Affine_FISH(
            self.params['W4'], self.params['b4'])

        self.last_layer = SigmoidWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            start = time.time()
            x = layer.forward(x)
            elapsed = time.time() - start
            # print(layer)
            # print('predict  taking  {0}'.format(elapsed) + '[sec]')

        return x

    def loss(self, x, t):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        start = time.time()
        # TODO: 時間かかってる
        y = self.predict(x)
        elapsed = time.time() - start
        # print('loss in sphere_convnet  taking  {0}'.format(elapsed) + '[sec]')
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=10):
# if t.ndim != 1:
# t = np.argmax(t, axis=1)

        acc = 0.0
        # print("Calculating Accuracy")
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)

            y = tf.convert_to_tensor(y, np.float32)
            tt = tf.convert_to_tensor(tt, np.float32)
            binary_acc = K.mean(K.equal(tt, K.round(y)), axis=-1)
            binary_acc = tf.Session().run(binary_acc)
            acc += np.sum(binary_acc)

# y = np.argmax(y, axis=1)
# acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）
        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        start = time.time()
        # TODO: ここで時間
        self.loss(x, t)
        elapsed = time.time() - start
        print('loss taking  {0}'.format(elapsed) + '[sec]')

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            start = time.time()
            dout = layer.backward(dout)
            elapsed = time.time() - start
# print(layer)
# print('backward in sphere_convnet  taking  {0}'.format(elapsed) + '[sec]')

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['gamma1'], grads['beta1'] = self.layers['Batch1'].dgamma, self.layers['Batch1'].dbeta
        grads['gamma2'], grads['beta2'] = self.layers['Batch2'].dgamma, self.layers['Batch2'].dbeta

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

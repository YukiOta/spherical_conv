# coding: utf-8
"""ユーティリティ的な関数たち
- convのアウトプットサイズを返す関数
- im2col
- col2im
- fish2col (つくる！)

"""

import numpy as np
import time
# import numba
from tqdm import tqdm
from multiprocessing import Pool
import fish2hemisphere_local as fish
# import fish2hemisphere_local as fish


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5]


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2 * pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):  # {{{
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    # input_data = np.random.randn(10, 3, 28, 28)  # 我が足した
    # input_data.shape  # 我が足した
    # filter_h, filter_w, stride, pad = (3, 3, 1, 1)  # 我が足した
    # img.shape
    # img[:, :, 0:28:1, 0:28:1].shape

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0),
                              (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    col.shape
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]  # }}}


# @numba.jit
def fish2col_diamond(input_data, level, filter_size=3, stride=1, pad=0):# {{{
    N, C, A, I, J = input_data.shape
    Q = 2**level
    filter_h, filter_w, stride = (filter_size, filter_size, 1)
    col = np.zeros((N, C, filter_h, filter_w, A, I, J))
    start = time.time()
    for y in range(filter_h):
        y_max = y + Q
        # print('y_max: ', y_max)
        for x in range(filter_w):
            x_max = x + 2 * Q
            # print('x_max: ', x_max)
            col[:, :, y, x, :, 1:-1, 1:-1] = \
                input_data[:, :, :, y:y_max:stride, x:x_max:stride]
    elapsed = time.time() - start
    print('fish2col in sphere_convnet  taking  {0}'.format(elapsed) + '[sec]')
    col = col.transpose(0, 4, 5, 6, 1, 2, 3).reshape(N * A * I * J, -1)
    return col# }}}


# @numba.jit
def col2fish_diamond(col, input_data, level, filter_size=3, stride=1):# {{{
    N, C, A, I, J = input_data.shape
    Q = 2**level
    filter_h, filter_w, stride = (filter_size, filter_size, 1)
    col = col.reshape(N, A, I, J, C, filter_h,
                      filter_w).transpose(0, 4, 5, 6, 1, 2, 3)
    img = np.zeros((N, C, A, I, J))
    start = time.time()
    for y in range(filter_h):
        y_max = y + Q
        for x in range(filter_w):
            x_max = x + 2 * Q
            img[:, :, :, y:y_max:stride, x:x_max:stride] += \
                col[:, :, y, x, :, 1:-1, 1:-1]
    elapsed = time.time() - start
    print('col2fish in sphere_convnet  taking  {0}'.format(elapsed) + '[sec]')
    return img# }}}


# @numba.jit
def pool_backward(col_dout, dmax, level=3, cell_num=7):
    '''
    poolingのbackwardの時に，アップサンプリングしながら，
    勾配を伝搬する
    input:
    col_dout = 伝搬してきた誤差． (data_num, FN*cell_num)
    level: 伝搬してきた配列のlevel
    dmax = 伝搬していく誤差の勾配

    return : array_out
    '''

    # print("backwarding in Poolin layer")
    # print(dmax.shape)
    data_num, FN, A, I, J = dmax.shape
    # print(data_num, FN, A, I, J)
    area_point_num = int(col_dout.shape[0] / data_num)
    # print(area_point_num)
    # zero定義
    zeros = np.zeros((FN))

    for data in range(data_num):
        out_tmp = col_dout[area_point_num * data: area_point_num * (data + 1)]
        # out_tmp = out_tmp.reshape(out_tmp.shape[0], cell_num, -1, order="F")
        out_tmp = out_tmp.reshape(out_tmp.shape[0], -1)
        # out_tmp = out_tmp[:, 0, :]
        # カウンターの設定
        count = 0
        array = out_tmp[count]
        count += 1
        for area in range(1, 6):
            for i_axis in range(1, I - 1):
                for j_axis in range(1, J - 1):
                    if i_axis % 2 == 0 and j_axis % 2 == 1:
                        array = np.vstack((array, out_tmp[count, 0::cell_num]))
                        count += 1
                    else:
                        array = np.vstack((array, zeros))
        array = np.vstack((array, out_tmp[count, 0::cell_num]))
        # print(array_1.shape)

    array_out = np.reshape(array, array.size, order='F')
    # print("array_out.shape: " + str(array_out.shape))
    return array_out


def loop_fish2col(k, input_data, level):  # {{{
    sphere_tmp = input_data[k]
    N, C, A, I, J = input_data.shape
    # sphere_tmp = input_data[0]
    array = 0
    Q = 2**level
    count = 0
    '''
    zenithについて追加する
    '''
    zenith_point = (1, 0, 1)
    _, neighbor = fish.find_neighbor(
        level=level,
        point=zenith_point,
        coord=True)
    a, i, j = zenith_point
    array = sphere_tmp[:, a, i, j]
    for idx, point in enumerate(neighbor):
        a, i, j = point
        array_0 = sphere_tmp[:, a, i, j]
        array = np.vstack((array, array_0))
    array = array.T.reshape(1, -1)
    if count == 0:
        array_full = array
        # print(array_full.shape)
        count += 1
    else:
        array_full = np.vstack((array_full, array))
    '''
    zenith add done
    '''
    for area in range(1, 6):
        for i_axis in range(1, Q + 1):
            # これは、半球のとき
            # j_end = np.int(3 * Q / 2 + 1 - i_axis)
            # こっちは全球
            j_end = 2 * Q
            for j_axis in range(1, j_end + 1):
                center_point = (area, i_axis, j_axis)
                # print(center_point)
                # a, i, j = center_point
                _, neighbor = fish.find_neighbor(
                    level=level,
                    point=center_point,
                    coord=True)
                array = sphere_tmp[:, area, i_axis, j_axis]
                for idx, point in enumerate(neighbor):
                    a, i, j = point
                    array_0 = sphere_tmp[:, a, i, j]
                    array = np.vstack((array, array_0))
                array = array.T.reshape(1, -1)
                array_full = np.vstack((array_full, array))
    '''
    nadirについて追加する
    '''
    nadir_point = (1, Q, 2 * Q + 1)
    _, neighbor = fish.find_neighbor(level=level,
                                     point=nadir_point,
                                     coord=True)
    a, i, j = nadir_point
    array = sphere_tmp[:, a, i, j]
    for idx, point in enumerate(neighbor):
        a, i, j = point
        array_0 = sphere_tmp[:, a, i, j]
        array = np.vstack((array, array_0))
    array = array.T.reshape(1, -1)
    array_full = np.vstack((array_full, array))
    '''
    nadir add done
    '''
    return array_full  # }}}


def wrapper_fish2col(args):
    return loop_fish2col(*args)


def fish2col(input_data, level, stride=1, pad=0):
    '''
    並列処理をする
    '''
    tutumimono = [[k, input_data, level] for k in range(len(input_data))]
    p = Pool(20)
    output = p.map(wrapper_fish2col, tutumimono)
    p.close()
    out_list = []
    append = out_list.append

    for i in range(len(output)):
        append(output[i])

    output = np.array(output)
    # print(output.shape)
    output = output.reshape(output.shape[0]*output.shape[1], -1)
    return output


def fish2col_(input_data, level, stride=1, pad=0):  # {{{
    """
    球面配列を，行列に変換する
    半球だけだったけど、全球に対応させる
    issue #1
    Parameters
    ----------
    input_data : (データ数, チャンネル, 領域，高さ, 幅)の5次元配列からなる入力データ
    stride : ストライド
    pad : パディング
    Returns
    -------
    arry_full : 2次元配列 (適用領域, FH*FW*C)
    """

    N, C, A, I, J = input_data.shape
    # sphere_tmp = input_data[0]
    array = 0
    Q = 2**level
    count = 0

    for data in tqdm(range(N)):
        sphere_tmp = input_data[data]
        '''
        zenithについて追加する
        '''
        zenith_point = (1, 0, 1)
        _, neighbor = fish.find_neighbor(
            level=level,
            point=zenith_point,
            coord=True)
        a, i, j = zenith_point
        array = sphere_tmp[:, a, i, j]
        for idx, point in enumerate(neighbor):
            a, i, j = point
            array_0 = sphere_tmp[:, a, i, j]
            array = np.vstack((array, array_0))
        array = array.T.reshape(1, -1)
        if count == 0:
            array_full = array
            # print(array_full.shape)
            count += 1
        else:
            array_full = np.vstack((array_full, array))
        '''
        zenith add done
        '''
        for area in range(1, 6):
            for i_axis in range(1, Q + 1):
                # これは、半球のとき
                # j_end = np.int(3 * Q / 2 + 1 - i_axis)
                # こっちは全球
                j_end = 2 * Q
                for j_axis in range(1, j_end + 1):
                    center_point = (area, i_axis, j_axis)
                    # print(center_point)
                    # a, i, j = center_point
                    _, neighbor = fish.find_neighbor(
                        level=level,
                        point=center_point,
                        coord=True)
                    array = sphere_tmp[:, area, i_axis, j_axis]
                    for idx, point in enumerate(neighbor):
                        a, i, j = point
                        array_0 = sphere_tmp[:, a, i, j]
                        array = np.vstack((array, array_0))
                    array = array.T.reshape(1, -1)
                    array_full = np.vstack((array_full, array))
        '''
        nadirについて追加する
        '''
        nadir_point = (1, Q, 2 * Q + 1)
        _, neighbor = fish.find_neighbor(level=level,
                                         point=nadir_point,
                                         coord=True)
        a, i, j = nadir_point
        array = sphere_tmp[:, a, i, j]
        for idx, point in enumerate(neighbor):
            a, i, j = point
            array_0 = sphere_tmp[:, a, i, j]
            array = np.vstack((array, array_0))
        array = array.T.reshape(1, -1)
        array_full = np.vstack((array_full, array))
        '''
        nadir add done
        '''
    # array_full = array_full.get()
    # print("fish2col output: " + str(array_full.shape))

    return array_full  # }}}


def loop_col2fish(k, data_num, input_mat, level, FN,
                  cell_num, backward):  # {{{
    data = k
    area_point_num = int(input_mat.shape[0] / data_num)
    Q = 2**level
    out_array = np.zeros((data_num, FN, 6, Q + 2, 2 * Q + 2))
    out_tmp = input_mat[area_point_num *
                        data:area_point_num * (data + 1)]
    if backward is True:
        out_tmp = out_tmp.reshape(out_tmp.shape[0], -1, FN)
    count = 0
    # add zenith
    for area in range(1, 6):
        if backward is True:
            out_array[data, :, area, 0, 1] = out_tmp[count, 0::cell_num]
        else:
            out_array[data, :, area, 0, 1] = out_tmp[count, :]
    count += 1
    for area in range(1, 6):
        for i_axis in range(1, Q + 1):
            j_end = 2 * Q
            # j_end = np.int(3 * Q / 2 + 1 - i_axis)
            for j_axis in range(1, j_end + 1):
                if backward is True:
                    out_array[data, :, area, i_axis,
                              j_axis] = out_tmp[count, 0::cell_num]
                else:
                    out_array[data, :, area, i_axis,
                              j_axis] = out_tmp[count, :]
                count += 1
    # add nadir
    for area in range(1, 6):
        if backward is True:
            out_array[data, :, area, Q, 2 * Q + 1] = \
                    out_tmp[count, 0::cell_num]
        else:
            out_array[data, :, area, Q, 2 * Q + 1] = out_tmp[count, :]
    count += 1
    return out_array  # }}}


def wrapper_col2fish(args):
    return loop_col2fish(*args)


def col2fish_(data_num, input_mat, level, FN=3,
             cell_num=7, pool=False, backward=False):
    '''
    並列処理をする
    '''
    if pool is True:
        print('pool is nonavailable')
    else:
        tutumimono = [[k, data_num, input_mat, level, FN, cell_num, backward] for k in tqdm(range(data_num))]
        p = Pool(20)
        output = p.map(wrapper_col2fish, tutumimono)
        p.close()
        out_list = []
        append = out_list.append

        output = np.array(output)
        # print(output.shape)
        # output = output.reshape(output.shape[0]*output.shape[1], -1)
        # print(output.shape)
        return output


# @numba.jit
def col2fish(data_num, input_mat, level, FN=3,
             cell_num=7, pool=False, backward=False):
    """
    行列を，球面配列に戻す
    全球バージョン
    issue #2
    Parameters
    ----------
    input_mat : (適用領域，FH*FW*C) pooling時は(適用領域*FN,)
    pool: Trueなら、level-1にダウンサイズされる
    arg: (適用領域*FN,)
    Returns
    -------
    out_array : (データ数, チャンネル, 領域，高さ, 幅)
    """

    if pool is True:
        """
        input_mat and argを、(適用領域、FN)に変形する
        """
        # print("col2fish with pooling mode")
        input_mat = input_mat.reshape(-1, FN)

        area_point_num = int(input_mat.shape[0] / data_num)
        Q = 2**(level - 1)
        out_array = np.zeros((data_num, FN, 6, Q + 2, 2 * Q + 2))
        for data in tqdm(range(data_num)):
            out_tmp = input_mat[area_point_num *
                                data:area_point_num * (data + 1)]
            # out_tmp.shape -> (91, 3) count = 0 for area in range(1, 6):
            count = 0
            out_array[data, :, area, 0, 1] = out_tmp[count, :]
            count += 1
            for area in range(1, 6):
                for i_axis in range(1, Q + 1):
                    if i_axis % 2 == 0:
                        # j_end = np.int(3 * Q / 2 + 1 - i_axis)
                        j_end = 2 * Q
                        for j_axis in range(1, j_end + 1):
                            if j_axis % 2 == 1:
                                # count += 1
                                out_array[data, :, area, i_axis // 2,
                                          (j_axis + 1) // 2] = out_tmp[count, :]
                                count += 1
                            else:
                                count += 1
                    else:
                        count += 1
            for area in range(1, 6):
                # print(out_tmp.shape)
                # print(count)
                out_array[data, :, area, Q, 2 * Q + 1] = out_tmp[count, :]
        return out_array

    else:
        area_point_num = int(input_mat.shape[0] / data_num)
        Q = 2**level
        out_array = np.zeros((data_num, FN, 6, Q + 2, 2 * Q + 2))
        for data in range(data_num):
            out_tmp = input_mat[area_point_num *
                                data:area_point_num * (data + 1)]
            if backward is True:
                out_tmp = out_tmp.reshape(out_tmp.shape[0], -1, FN)
            count = 0
            # add zenith
            for area in range(1, 6):
                if backward is True:
                    out_array[data, :, area, 0, 1] = out_tmp[count, 0::cell_num]
                else:
                    out_array[data, :, area, 0, 1] = out_tmp[count, :]
            count += 1
            for area in range(1, 6):
                for i_axis in range(1, Q + 1):
                    j_end = 2 * Q
                    # j_end = np.int(3 * Q / 2 + 1 - i_axis)
                    for j_axis in range(1, j_end + 1):
                        if backward is True:
                            out_array[data, :, area, i_axis,
                                    j_axis] = out_tmp[count, 0::cell_num]
                        else:
                            out_array[data, :, area, i_axis,
                                    j_axis] = out_tmp[count, :]
                        count += 1
            # add nadir
            for area in range(1, 6):
                if backward is True:
                    out_array[data, :, area, Q, 2 * Q + 1] = out_tmp[count, 0::cell_num]
                else:
                    out_array[data, :, area, Q, 2 * Q + 1] = out_tmp[count, :]
            count += 1
        return out_array

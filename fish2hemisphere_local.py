# coding: utf-8
""" sphere_conv用のfish2hemisphere.py

"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

SAVE_DIR = "./RESULT/spherical_array/"
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def spherical2cartesian(coord, r=1):
    """球座標をデカルト座標系に変換
    Parameters
    ----------
    coord[0] : 緯度
    coord[1] : 経度

    Retrun
    ----------
    x, y, z : 各座標
    """
    col = coord[0]
    lon = coord[1]
    x = r * np.sin(col) * np.cos(lon)
    y = r * np.sin(col) * np.sin(lon)
    z = r * np.cos(col)
    return np.array([x, y, z])


def scatter3d(filename, sphere=True, view=(), nb_point=None, r=1):
    """3次元プロット
    Parameters
    ----------
    filename : 保存するfilename
    sphere : Trueにすると，北半球のグリッドを表示する
    view : 球面を見る角度を決められる
    nb_point : 指定すれば，指定した点の近傍を色付けして出力
    """
    ele, azi = view
    figure = plt.figure(figsize=(8, 8))
    ax = Axes3D(figure)
    # ax = figure.add_subplot(111, projection='3d')
    ax.set_aspect("equal")

    if sphere is True:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_wireframe(x, y, z, color='r', linewidth=0.5)

    cartesian_vertex = np.array([
        [spherical2cartesian(c, r=r) for c in point_vertex]
    ])
    cartesian_coords = np.array([
        [spherical2cartesian(c, r=r) for c in points]
    ])

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_xlim(-1.1*r, 1.1*r)
    ax.set_ylim(-1.1*r, 1.1*r)
    ax.set_zlim(-1.1*r, 1.1*r)
    ax.scatter3D(
        xs=cartesian_coords.T[0],
        ys=cartesian_coords.T[1],
        zs=cartesian_coords.T[2],
        linewidth=1,
        color='k',
        )
    ax.scatter3D(
        xs=cartesian_vertex.T[0],
        ys=cartesian_vertex.T[1],
        zs=cartesian_vertex.T[2],
        linewidth=5,
        color='b',
        )
    if nb_point is not None:
        area, j_cen, k_cen = nb_point
        neighbor = find_neighbor(point=nb_point)
        cartesian_neighbor = np.array([
            [spherical2cartesian(c, r=r) for c in neighbor]
        ])
        cartesian_center = spherical2cartesian(pos[area][j_cen][k_cen], r=r)
        ax.scatter3D(
            xs=cartesian_center[0],
            ys=cartesian_center[1],
            zs=cartesian_center[2],
            linewidth=8,
            color='m',
            marker='h'
            )
        ax.scatter3D(
            xs=cartesian_neighbor.T[0],
            ys=cartesian_neighbor.T[1],
            zs=cartesian_neighbor.T[2],
            linewidth=8,
            color='g',
            marker='*',
            alpha=1
            )

    ax.view_init(elev=ele, azim=azi)
    plt.savefig(SAVE_DIR+filename)
    plt.show()
    plt.close()
    return


def cal_angle(pos):
    """角度計算する関数
    正二十面体の各頂点を計算するときに使ってる
    """
    theta = pos[0]
    # phi = (kai + pos[1]) % (2 * np.pi)
    phi = (kai + pos[1])
    return (theta, phi)


def cal_mid(area=1, coord1=(0, 0), coord2=(1, 1)):
    """中点を計算するプログラム
    Parameters
    ----------

    """

    j_1, k_1 = coord1
    j_2, k_2 = coord2
    j = int((j_1 + j_2) / 2)
    k = int((k_1 + k_2) / 2)

    if coord1 == (0, 1):
        mid_0 = (pos[area][j_1][k_1][0] + pos[area][j_2][k_2][0]) / 2
        mid_1 = pos[area][j_2][k_2][1]
        mid_point = (mid_0, mid_1)
    elif coord2 == (0, 1):
        mid_0 = (pos[area][j_1][k_1][0] + pos[area][j_2][k_2][0]) / 2
        mid_1 = pos[area][j_1][k_1][1]
        mid_point = (mid_0, mid_1)
    else:
        mid_point = (pos[area][j_1][k_1] + pos[area][j_2][k_2]) / 2
    return (j, k, mid_point)


def cal_unit_12(area=1, coord1=(0, 0), coord2=(1, 1)):
    """1 -> 2の方向への単位ベクトルを計算する関数
    vertexの間の単位ベクトルを計算するときに使っているっぽい？
    Parameters
    ----------
    """
    j_1, k_1 = coord1
    j_2, k_2 = coord2
    # j = int((j_1 + j_2) / 2)
    # k = int((k_1 + k_2) / 2)

    if coord1 == (0, 1):
        start = pos[area][j_1][k_1][0]
        end = pos[area][j_2][k_2][0]
        edge_unit = np.array(((end - start) / Q, 0))
    elif coord2 == (Q, 2*Q+1):
        start = pos[area][j_1][k_1][0]
        end = pos[area][j_2][k_2][0]
        edge_unit = np.array(((end - start) / Q, 0))
    else:
        start = pos[area][j_1][k_1]
        end = pos[area][j_2][k_2]
        edge_unit = (end - start) / Q

    return edge_unit


def cal_midunit_12(area=1, coord1=(0, 0), coord2=(1, 1), sep=2):
    """真ん中の部分での単位ベクトル？？
    cal_unit_12との違いは，パラメータにsepを含む子こと．
    使用時は，for文でsepの値を指定しているから，こっちを使っている
    Parameters
    ----------
    area : どの平行四辺形か (1-5)
    coord1 : 始点
    coord2 : 終点
    sep : 何分割するかのパラメータ
    """
    j_1, k_1 = coord1
    j_2, k_2 = coord2
    # j = int((j_1 + j_2) / 2)
    # k = int((k_1 + k_2) / 2)

    if coord1 == (0, 1):
        start = pos[area][j_1][k_1][0]
        end = pos[area][j_2][k_2][0]
        edge_unit = np.array(((end - start) / sep, 0))
    elif coord2 == (Q, 2*Q+1):
        start = pos[area][j_1][k_1][0]
        end = pos[area][j_2][k_2][0]
        edge_unit = np.array(((end - start) / sep, 0))
    else:
        start = pos[area][j_1][k_1]
        end = pos[area][j_2][k_2]
        edge_unit = (end - start) / sep

    return edge_unit


def tessellation(
        area=1, a=(0, 1), b=(4, 1), c=(0, 5),
        d=(0, 5), e=(4, 5), f=(0, 9)):
    """球面を分割する関数
    Parameters
    -----------
    area : 平行四辺形の番号 (1~5)
    a - f : 平行四辺形を最初に三角形で分割したときの，拡張点
    """
    j_a, k_a = a
    j_b, k_b = b
    j_c, k_c = c
    j_d, k_d = d
    j_e, k_e = e
    j_f, k_f = f

    unit_ab = cal_unit_12(area=area, coord1=(j_a, k_a), coord2=(j_b, k_b))
    unit_ac = cal_unit_12(area=area, coord1=(j_a, k_a), coord2=(j_c, k_c))
    unit_bc = cal_unit_12(area=area, coord1=(j_b, k_b), coord2=(j_c, k_c))
    unit_cd = cal_unit_12(area=area, coord1=(j_c, k_c), coord2=(j_d, k_d))
    unit_ce = cal_unit_12(area=area, coord1=(j_c, k_c), coord2=(j_e, k_e))
    unit_de = cal_unit_12(area=area, coord1=(j_d, k_d), coord2=(j_e, k_e))
    unit_bd = cal_unit_12(area=area, coord1=(j_b, k_b), coord2=(j_d, k_d))
    unit_df = cal_unit_12(area=area, coord1=(j_d, k_d), coord2=(j_f, k_f))
    unit_ef = cal_unit_12(area=area, coord1=(j_e, k_e), coord2=(j_f, k_f))

    for i in range(1, Q):
        # ABC
        pos[area][i][k_a] = np.array(
            (pos[area][j_a][k_a][0], pos[area][j_b][k_b][1])) + i * unit_ab
        pos[area][j_a][k_a+i] = np.array(
            (pos[area][j_a][k_a][0], pos[area][j_c][k_c][1])) + i * unit_ac
        pos[area][j_b-i][k_b+i] = pos[area][j_b][k_b] + i * unit_bc
        # CDE
        pos[area][i][k_c] = pos[area][j_c][k_c] + i * unit_cd
        pos[area][j_c][k_c+i] = pos[area][j_c][k_c] + i * unit_ce
        pos[area][j_d-i][k_d+i] = pos[area][j_d][k_d] + i * unit_de
        # BD
        pos[area][j_b][k_b+i] = pos[area][j_b][k_b] + i * unit_bd
        # DEF
        pos[area][j_d][k_d+i] = pos[area][j_d][k_d] + i * unit_df
        pos[area][j_e+i][k_e] = pos[area][j_e][k_e] + i * unit_ef

        # check_hemisphere(pos=pos[area][i][k_a], p_list=points)
        # check_hemisphere(pos=pos[area][j_a][k_a+i], p_list=points)
        # check_hemisphere(pos=pos[area][j_b-i][k_b+i], p_list=points)
        #
        # check_hemisphere(pos=pos[area][i][k_c], p_list=points)
        # check_hemisphere(pos=pos[area][j_c][k_c+i], p_list=points)
        # check_hemisphere(pos=pos[area][j_d-i][k_d+i], p_list=points)
        #
        # check_hemisphere(pos=pos[area][j_b][k_b+i], p_list=points)
        # check_hemisphere(pos=pos[area][j_d][k_d+i], p_list=points)
        # check_hemisphere(pos=pos[area][j_e+i][k_e], p_list=points)
        check_hemisphere(
            pos=pos, point=(area, i, k_a), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )
        check_hemisphere(
            pos=pos, point=(area, j_a, k_a+i), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )

        check_hemisphere(
            pos=pos, point=(area, j_b-i, k_b+i), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )

        check_hemisphere(
            pos=pos, point=(area, i, k_c), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )
        check_hemisphere(
            pos=pos, point=(area, j_c, k_c+i), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )
        check_hemisphere(
            pos=pos, point=(area, j_d-i, k_d+i), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )

        check_hemisphere(
            pos=pos, point=(area, j_b, k_b+i), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )
        check_hemisphere(
            pos=pos, point=(area, j_d, k_d+i), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )
        check_hemisphere(
            pos=pos, point=(area, j_e+i, k_e), p_list=points,
            img_sphere=img_sphere, make_img_array=True
        )

    # make points points inside of triangles
    # from left to right, horizontal direction
    for i in range(1, Q):
        if i >= 2:
            unit_abc = cal_midunit_12(area=area, coord1=(i, k_a), coord2=(j_a, k_a+i), sep=i)
            unit_cde = cal_midunit_12(area=area, coord1=(i, k_c), coord2=(j_c, k_c+i), sep=i)
            for j in range(1, i):
                pos[area][i-j][k_a+j] = pos[area][i][k_a] + j * unit_abc
                pos[area][i-j][k_c+j] = pos[area][i][k_c] + j * unit_cde

                check_hemisphere(
                    pos=pos, point=(area, i-j, k_a+j), p_list=points,
                    img_sphere=img_sphere, make_img_array=True
                )
                check_hemisphere(
                    pos=pos, point=(area, i-j, k_c+j), p_list=points,
                    img_sphere=img_sphere, make_img_array=True
                )

                # check_hemisphere(pos=pos[area][i-j][k_a+j], p_list=points)
                # check_hemisphere(pos=pos[area][i-j][k_c+j], p_list=points)

        unit_bcd = cal_midunit_12(area=area, coord1=(j_b, k_b+i), coord2=(j_c+i, k_c), sep=Q-i)
        unit_def = cal_midunit_12(area=area, coord1=(j_d, k_d+i), coord2=(j_e+i, k_e), sep=Q-i)
        for j in range(1, Q-i):
            pos[area][j_b-j][k_b+i+j] = pos[area][j_b][k_b+i] + j * unit_bcd
            pos[area][j_d-j][k_d+i+j] = pos[area][j_d][k_d+i] + j * unit_def
            check_hemisphere(
                pos=pos, point=(area, j_b-j, k_b+i+j), p_list=points,
                img_sphere=img_sphere, make_img_array=True
            )
            check_hemisphere(
                pos=pos, point=(area, j_d-j, k_d+i+j), p_list=points,
                img_sphere=img_sphere, make_img_array=True
            )

            # check_hemisphere(pos=pos[area][j_b-j][k_b+i+j], p_list=points)
            # check_hemisphere(pos=pos[area][j_d-j][k_d+i+j], p_list=points)

    return


def check_hemisphere(pos, point, p_list, img_sphere, make_img_array=True):
    """半球に抑えこむためのチェッカー
    pos[0]には緯度の情報が入ってる

    一緒に，画像も作ってしまう

    Parameters
    ----------
    pos : 球面配列が入ったリスト
    point : (area, j, k)
    p_list : 北半球にあれば，p_listにその座標を追加する
    img_sphere : 最初の方で定義しておく，画素情報を入れ込む配列
    """
    area, j__, k__ = point
    channels = 3
    '''check sphereにしちゃう
    '''
    if pos[area][j__][k__][0] <= np.pi:
        p_list.append(pos[area][j__][k__])
        if make_img_array is True:
            for channel in range(channels):
                img_sphere[channel][area, j__, k__] = cal_value_from3d(
                    channel=channel,
                    position=(area, j__, k__),
                    r_pic=r_pic,
                    im_in=im_in,
                    pos_3d=pos
                )
        return


def find_neighbor(level, point=(1, 1, 2), coord=False):
    """注目点の近傍を持ってくる
    Parameters
    ----------
    point : 注目する点 (area, j ,k)

    return
    ----------
    neighbor : neighborの曲座標
    neighbor_coord : neigborの球面座標
    """
    Q = 2**level
    # print('in find_neighbor')
    # print(Q)
    area, j_p, k_p = check_neighbor(level=level, point=point)
    neighbor = []
    neighbor_coord = []
    neighbor_list = [
        (area, j_p-1, k_p+1),
        (area, j_p-1, k_p),
        (area, j_p, k_p-1),
        (area, j_p+1, k_p-1),
        (area, j_p+1, k_p),
        (area, j_p, k_p+1)
    ]
    if (j_p, k_p) == (0, 1):  # zenith
        for i in range(1, 6):
            neighbor.append(pos[i][1][1])
            neighbor_coord.append((i, 1, 1))
        '''zenithのとき、真ん中の値を足す
        '''
        neighbor.append(pos[1][0][1])
        neighbor_coord.append((1, 0, 1))
    elif (j_p, k_p) == (Q, 2*Q+1):  # nadir
        for i in range(1, 6):
            neighbor.append(pos[i][Q][2*Q])
            neighbor_coord.append((i, Q, 2*Q))
        '''nadirのときも、真ん中値を足す
        '''
        neighbor.append(pos[1][Q][2*Q+1])
        neighbor_coord.append((1, j_p, k_p))
    else:
        for point in neighbor_list:
            area, j_nb, k_nb = point
            area, j_nb, k_nb = check_neighbor(level=level, point=(area, j_nb, k_nb))
            neighbor.append(pos[area][j_nb][k_nb])
            neighbor_coord.append((area, j_nb, k_nb))

    if coord is True:
        return neighbor, neighbor_coord
    else:
        return neighbor


def check_neighbor(level, point=(1, 2, 1)):
    """近傍のチェッカー
    Parameters
    ----------
    point : 注目する点 (area, j, k)
    """
    Q = 2**level
    # print('in check_neighbor')
    # print(Q)
    area, j, k = point
    if k == 0:
        area = area - 1
        if area == 0:
            area = 5
        return (area, 1, j)

    elif j == Q+1:
        area = area - 1
        if area == 0:
            area = 5
        if k <= Q:
            return (area, 1, Q+k)
        else:
            return (area, k-Q, 2*Q)

    elif j == 0:
        area = area + 1
        if area == 6:
            area = 1
        if k >= 2 and k <= Q:
            return (area, k-1, 1)
        elif k >= Q+1 and k <= 2*Q+1:
            return (area, Q, k-Q)
        else:
            return (area, j, k)

    elif k == 2*Q+1:
        area = area + 1
        if area == 6:
            area = 1
        return (area, Q, j+Q+1)

    else:
        return (area, j, k)


def make_2d_array(pos_3d, Q):
    """3次元配列を2次元配列に変換する
    Parameters
    ----------
    pos_3d : area, j, k で構成されているリスト
    Q : 分割数
    """

    R_L = 3*Q
    i_2d, j_2d = (3*Q, 5*Q)
    pos_2d = np.zeros((i_2d+1, j_2d+1, 2))
    pos_2d_coord = np.zeros((i_2d+1, j_2d+1, 3), dtype=np.int)

    for i in range(R_L+1):
        if i == 0:  # 1
            pos_2d[i][0] = pos_3d[1][0][1]
            pos_2d_coord[i][0] = (1, 0, 1)
        elif i <= Q:  # 2
            j_p, k_p = (i, 1)
            for j in range(5*Q):
                point = j % i
                if j <= i-1:
                    pos_2d[i][j] = pos_3d[1][j_p-point][k_p+point]
                elif j <= 2*i-1:
                    pos_2d[i][j] = pos_3d[2][j_p-point][k_p+point]
                elif j <= 3*i-1:
                    pos_2d[i][j] = pos_3d[3][j_p-point][k_p+point]
                elif j <= 4*i-1:
                    pos_2d[i][j] = pos_3d[4][j_p-point][k_p+point]
                elif j <= 5*i-1:
                    pos_2d[i][j] = pos_3d[5][j_p-point][k_p+point]
        elif i <= Q*2:  # 3
            i_n = i - Q
            j_p, k_p = (Q, 1+i_n)
            for j in range(5*Q):
                point = j % Q
                if j <= Q-1:
                    pos_2d[i][j] = pos_3d[1][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (1, j_p-point, k_p+point)
                elif j <= 2*Q-1:
                    pos_2d[i][j] = pos_3d[2][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (2, j_p-point, k_p+point)
                elif j <= 3*Q-1:
                    pos_2d[i][j] = pos_3d[3][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (3, j_p-point, k_p+point)
                elif j <= 4*Q-1:
                    pos_2d[i][j] = pos_3d[4][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (4, j_p-point, k_p+point)
                elif j <= 5*Q-1:
                    pos_2d[i][j] = pos_3d[5][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (5, j_p-point, k_p+point)

        elif i < R_L:
            i_ = R_L - i
            j_p, k_p = (Q, 2*Q-i_)
            for j in range(5*i_):
                point = j % i_
                if j <= i_-1:
                    pos_2d[i][j] = pos_3d[1][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (1, j_p-point, k_p+point)
                elif j <= 2*i_-1:
                    pos_2d[i][j] = pos_3d[2][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (2, j_p-point, k_p+point)
                elif j <= 3*i_-1:
                    pos_2d[i][j] = pos_3d[3][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (3, j_p-point, k_p+point)
                elif j <= 4*i_-1:
                    pos_2d[i][j] = pos_3d[4][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (4, j_p-point, k_p+point)
                elif j <= 5*i_-1:
                    pos_2d[i][j] = pos_3d[5][j_p-point][k_p+point]
                    pos_2d_coord[i][j] = (5, j_p-point, k_p+point)
        elif i == R_L:
            pos_2d[i][0] = pos_3d[1][Q][2*Q+1]
            pos_2d_coord[i][0] = (1, Q, 2*Q+1)

    return pos_2d, pos_2d_coord


def make_2d_array_interpolated(pos_3d, Q):
    """3次元配列を2次元配列に変換する (for シータ・ファイ画像)
    theta_phi画像を作る為に，天頂付近の座標は，細かく分割する．
    Parameters
    ----------
    pos_3d : area, j, k で構成されているリスト
    Q : 分割数
    """

    R_L = 3*Q
    i_2d, j_2d = (3*Q, 5*Q)
    pos_2d = np.zeros((i_2d, j_2d, 2))
    pos_2d_coord = np.zeros((i_2d, j_2d, 3), dtype=np.int)
    # phi_delta = (2*np.pi) / (5*Q)
    tmp = np.linspace(0, 2*np.pi, 5*Q, endpoint=True)

    for i in range(R_L+1):
        if i == 0:  # 1
            pos_2d[i] = pos_3d[1][0][1]
            pos_2d_coord[i] = (1, 0, 1)
        elif i <= Q:  # 2
            theta = pos[1][i][1][0]
            pos_2d[i, :, 0] = theta
            pos_2d[i, :, 1] = tmp
        elif i <= Q*2:  # 3
            i_n = i - Q
            j_p, k_p = (Q, 1+i_n)
            if pos[1][j_p][k_p][0] <= np.pi/2:
                theta = pos[1][j_p][k_p][0]
                pos_2d[i, :, 0] = theta
                pos_2d[i, :, 1] = tmp

    return pos_2d


def make_2d_picture(pos_2d, r_pic, Q, im_in):

    cols, rows = (pos_2d.shape[1], pos_2d.shape[0])
    im_out = np.zeros((rows, cols, 3), np.uint8)

    R_L = 3*Q
    for channel in range(3):
        for i in range(R_L+1):
            if i == 0:  # 1
                for j in range(5):
                    im_out[i, j, channel] = cal_value(channel, position=(i, 0), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i <= Q:  # 2
                for j in range(5*i):
                    im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i <= Q*2:  # 3
                if pos_2d[i][0][0] <= np.pi/2:
                    for j in range(5*Q):
                        im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i < R_L:
                if pos_2d[i][0][0] <= np.pi/2:
                    i_ = R_L - i
                    for j in range(5*i_):
                        im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i == R_L:
                if pos_2d[i][0][0] <= np.pi/2:
                    im_out[i, 0, channel] = cal_value(channel, position=(i, 0), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)

    return im_out[:int(im_out.shape[0]/2), :, :]


def make_2d_theta_phi_picture(pos_2d, r_pic, Q, im_in):

    cols, rows = (pos_2d.shape[1], pos_2d.shape[0])
    im_out = np.zeros((rows, cols, 3), np.uint8)

    R_L = 3*Q
    for channel in range(3):
        for i in range(rows):
            for j in range(cols):
                im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)

    return im_out[:int(im_out.shape[0]/2), :, :]


def make_2d_center_picture(pos_2d, r_pic, Q, im_in):

    cols, rows = (pos_2d.shape[1], pos_2d.shape[0])
    im_out = np.zeros((rows, cols, 3), np.uint8)

    R_L = 3*Q
    for channel in range(3):
        for i in range(R_L+1):
            if i == 0:  # 1
                index = int(5*Q / 2 - 1)
                im_out[i, index, channel] = cal_value(channel, position=(i, 0), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i <= Q:  # 2
                if i % 2 == 0:
                    index = int((5*Q - 5*i) / 2)
                else:
                    index = int((5*Q - 5*i - 1) / 2)
                for j in range(5*i):
                    im_out[i, index+j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i <= Q*2:  # 3
                if pos_2d[i][0][0] <= np.pi/2:
                    for j in range(5*Q):
                        im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i < R_L:
                if pos_2d[i][0][0] <= np.pi/2:
                    i_ = R_L - i
                    for j in range(5*i_):
                        im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i == R_L:
                if pos_2d[i][0][0] <= np.pi/2:
                    im_out[i, 0, channel] = cal_value(channel, position=(i, 0), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)

    return im_out


def make_2d_round_picture(pos_2d, r_pic, Q, im_in):
    """画像に途切れる部分が内容に，画素値を，行において繰り返す関数
    Parameters
    ----------
    """
    cols, rows = (pos_2d.shape[1], pos_2d.shape[0])
    im_out = np.zeros((rows, cols-1, 3), np.uint8)

    R_L = 3*Q
    for channel in range(3):
        for i in range(R_L+1):
            value_tmp = []
            if i == 0:  # 1
                im_out[i, :, channel] = cal_value(channel, position=(i, 0), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i <= Q:  # 2
                if i % 2 == 0:
                    index = int((5*Q - 5*i) / 2)
                else:
                    index = int((5*Q - 5*i - 1) / 2)
                for j in range(5*i):
                    tmp = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
                    value_tmp.append(tmp)
                for k in range(5*i+index+1):
                    if index+k < 80:
                        im_out[i, index+k, channel] = value_tmp[k % (5*i)]
                    if index-k >= 0:
                        im_out[i, index-k, channel] = value_tmp[4-((k-1) % (5*i))]
            elif i <= Q*2:  # 3
                if pos_2d[i][0][0] <= np.pi/2:
                    for j in range(5*Q):
                        im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i < R_L:
                if pos_2d[i][0][0] <= np.pi/2:
                    i_ = R_L - i
                    for j in range(5*i_):
                        im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i == R_L:
                if pos_2d[i][0][0] <= np.pi/2:
                    im_out[i, 0, channel] = cal_value(channel, position=(i, 0), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)

    return im_out[:int(im_out.shape[0]/2), :, :]


def make_2d_neighbor_considered_picture(pos_2d, pos_2d_coord, r_pic, Q, im_in):
    """画像に途切れる部分が内容に，find_neighborを用いて，一つ下の列から
    画素値を補間
    Parameters
    ----------
    pos_2d_coord : 2次元配列の座標に対する，(area, i, j)が格納されてる
    neib_coord[1] : 対象とするpointの右上の点
    """
    cols, rows = (pos_2d.shape[1], pos_2d.shape[0])
    im_out = np.zeros((rows, cols, 3), np.uint8)

    for channel in range(3):
        for i in reversed(range(Q*2+1)):
            if i == 0:  # 1
                im_out[i, :, channel] = cal_value(channel, position=(i, 0), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
            elif i <= Q:  # 2
                for j in range(5*Q):
                    neib_pos, neib_coord = find_neighbor(point=pos_2d_coord[i+1, j], coord=True, level=level)
                    area_, i_, j_ = neib_coord[1]
                    pos_2d_coord[i, j] = neib_coord[1]
                    im_out[i, j, channel] = img_sphere[channel, area_, i_, j_]
            elif i <= Q*2:  # 3
                if pos_2d[i][0][0] <= np.pi/2:
                    for j in range(5*Q):
                        im_out[i, j, channel] = cal_value(channel, position=(i, j), r_pic=r_pic, im_in=im_in, pos_2d=pos_2d)
    return im_out[:int(im_out.shape[0]/2), :, :]



def cal_value(channel, position, r_pic, im_in, pos_2d):

    channel = channel
    i_2d, j_2d = position
    x_f, y_f = sphere2fish(pos_2d[i_2d][j_2d], r_pic)
    # im_out[i, j, channel] = bilinear_interpolate(im_in[:, :, channel], x, y)
    value = bilinear_interpolate(im_in[:, :, channel], x_f, y_f)

    return value


def cal_value_from3d(channel, position, r_pic, im_in, pos_3d):
    """球面配列から，その位置における画素値をバイリニア補間で直接計算する．
    Parameters
    ----------
    channel : R, G, B チャンネルのこと (R:0, G:1, B:2)
    position : 球面配列における座標．(area, i, j)
    r_pic : 魚眼画像の画像部分の半径
    im_in : 魚眼画像のnparray
    pos_3d : 球面配列 Numpy配列orリスト (area, i, j, 2)みたいな配列

    Return
    ---------
    指定されたchannelにおける画素値
    """
    channel = channel
    area, i_3d, j_3d = position
    x_f, y_f = sphere2fish(pos_3d[area][i_3d][j_3d], r_pic)
    # im_out[i, j, channel] = bilinear_interpolate(im_in[:, :, channel], x, y)
    value = bilinear_interpolate(im_in[:, :, channel], x_f, y_f)

    return value


def sphere2fish(sphere_point, r_pic):

    theta, phi = sphere_point
    r_f = r_pic * np.tan(theta/2)
    # r_f = r_pic * np.sin(theta)
    # r_f = 2 * r_pic * theta / np.pi
    # r_f = np.sqrt(2) * r_pic * np.sin(theta/2)
    x_f = r_f * np.cos(phi) + r_pic
    y_f = r_f * np.sin(phi) + r_pic
    return np.array([x_f, y_f])


def bilinear_interpolate(im, x, y):
    """バイナリ補間
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def make_considered_picture(img, level, return_array=0):
    """Parameters
    r : 分割レベル
    Q : 平行四辺形の短辺における点の数．座標にもなる．
    i : 平行四辺形 (area) の数 (に，実装の都合上+1してる)
    j : 平行四辺形の短辺の座標の上限 (に，実装の都合上+1してる)
    k : 平行四辺形の長辺の座標の上限 (に，実装の都合上+1してる)
    kai : bc方向(円周方向)の単位角度 (論文参照)
    tau : ab方向(天頂から天底方向)の単位角度
    img_sphere : 球面配列に画素値が入るような配列 (3, i, j, k)
    """
    global r_pic, im_in, r, Q, i, j, k, kai, tau, img_sphere, pos, points
    global point_vertex
    r = level
    Q = 2**r
    i = 5 + 1
    j = Q + 1 + 1
    k = 2*Q + 1 + 1
    kai = 2 * np.pi / 5
    tau = np.arctan(2)
    img_sphere = np.zeros((3, i, j, k))

    """画像の読み込み
    PILを使ってImage.open
    ここでは計算の不可を少なくするため，(100, 100)にリサイズしている

    r_pic : 魚眼画像の半径 (画像がある部分)
    """
    # im_in = Image.open("../1data/test_image/cloud_half.png")
    # im_in = im_in.resize((100, 100))
    # im_in = np.array(im_in)
    im_in = img
    r_pic = int(im_in.shape[0]/2)


    """球面座標を保持するリストの定義
    pos[area][短辺座標][長辺座標]

    Parameters
    ----------
    pos[i][j][k][0] : ido theta [0, pi]
    pos[i][j][k][1] : keido phi [0, 2*pi]
    """
    pos = np.zeros((i, j, k, 2))
    pos[1][0][1] = (0, 0)  # a
    pos[1][Q][1] = (tau, 0)  # b
    pos[1][0][Q+1] = (tau, kai)  # c
    pos[1][Q][Q+1] = (np.pi - tau, kai / 2)  # d
    pos[1][0][2*Q+1] = (np.pi - tau, kai * 3 / 2)  # e
    pos[1][Q][2*Q+1] = (np.pi, 0)  # f

    """点のリスト
    points : 分割された点を放り込んでいく
    point_vertex : 頂点の座標を放り込んでいく
    vertex_list : area1の球面座標は手打ちでいれている (初期値的なね)
    """
    points = []
    point_vertex = []
    vertex_list = [
        (1, 0, 1),
        (1, Q, 1),
        (1, 0, Q+1),
        (1, Q, Q+1),
        (1, 0, 2*Q+1),
        (1, Q, 2*Q+1),
    ]

    """北半球にある点群だけを取り出してくる
    # vertex : こっちでは，area1を計算してる
    # ALL vertexes of icosahedron : こっちでは，全ての面を計算する
    """
    # vertex
    for point in vertex_list:
        area, j_v, k_v = point
        check_hemisphere(
            pos=pos,
            point=(area, j_v, k_v), p_list=point_vertex,
            img_sphere=img_sphere, make_img_array=True
        )

    # ALL vertexes of icosahedron
    for point in vertex_list:
        area, j_v, k_v = point
        for i in range(1, 5):
            pos[i+1][j_v][k_v] = cal_angle(pos[i][j_v][k_v])
            check_hemisphere(
                pos=pos,
                point=(i+1, j_v, k_v), p_list=point_vertex,
                img_sphere=img_sphere, make_img_array=True
            )

    """平行四辺形の分割
    基本的に，areaの値以外変えていない
    Parameters
    ----------
    area : どの平行四辺形か
    a-f : 頂点の座標を指定する

    returnは無いが，暗示的に先ほど定義したpointsリストに座標を放り込んでいる
    """
    for i in range(1, 6):
        tessellation(area=i, a=(0, 1), b=(Q, 1), c=(0, Q+1), d=(Q, Q+1), e=(0, 2*Q+1), f=(Q, 2*Q+1))
        # print(len(points))

    """これは実験的な，area1のみの分割に対応する
    """
    # tessellation(area=1, a=(0, 1), b=(Q, 1), c=(0, Q+1), d=(Q, Q+1), e=(0, 2*Q+1), f=(Q, 2*Q+1))

    """点の可視化
    Parameters
    ----------
    filename : 保存するfilenameを指定できる
    view=(a, b) : どの角度で見るか
    area : area
    nb_point=(area, i, j) : 指定すれば，指定した点の周りの近傍を色付けして出力する
    r : 魚眼画像の円部分の半径である．
    """
    a, b, area, i, j = (30, 30, 1, Q, 1)
    # scatter3d(filename='hemi_%d_%d_%d_%d%d%d.png' % (r, a, b, area, i, j), view=(a, b), nb_point=(area, i, j), r=r_pic)
    # scatter3d(filename='hemi_%d_%d_%d_%d%d%d.png' % (r, a, b, area, i, j), view=(a, b), nb_point=None, r=r_pic)

    """球面上に配置した点の座標から，球面画像の作成
    """
    pos_2d, pos_2d_coord = make_2d_array(pos_3d=pos, Q=Q)
    # pos_2d = make_2d_array_interpolated(pos_3d=pos, Q=Q)
    if return_array == 1:
        return img_sphere, pos
    # im = make_2d_picture(pos_2d=pos_2d, r_pic=r_pic, Q=Q, im_in=im_in)
    # im_center = make_2d_center_picture(pos_2d=pos_2d, r_pic=r_pic, Q=Q, im_in=im_in)
    # im_consider = make_2d_neighbor_considered_picture(pos_2d_coord=pos_2d_coord, pos_2d=pos_2d, r_pic=r_pic, Q=Q, im_in=im_in, level=level)

    return im_consider

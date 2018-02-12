# coding: utf-8
import numpy as np
import fish2hemisphere_local as fish
# import utilities as ut
from PIL import Image
# import cv2 as cv
import sys
from tqdm import tqdm


def pers2equi2(input_image, start, base=(0, 0), hemisphere=200, r=1):
    '''perspective imageをequirectangularに
    プログラムが間違っていたから，`pers2equi`の改良版

    ------------
    hemisphere: 半球あたりのピクセル数
    r:ここでは，焦点距離
    input_image: 入力となるperspective画像, numpy配列
    注意！np.asarrayとするときに、floatを指定してはいけない。
    start: どの角度で見るか．3~5

    return
    -------------
    array: equirectangular画像[0, 255]
    '''
    img_ori = input_image

    # 見る角度の指定
    theta_v = np.pi * start
# if theta_v < 0.000001:
    # theta_v = 0.0
    # theta_v = 0
    phi_v = np.pi * r
# if phi_v < 0.000001:
    # phi_v = 0.0

    # 透視画像としての，ピクセル数
    n_x = img_ori.shape[0]
    n_y = img_ori.shape[1]

    # 視野角と，焦点距離の指定
    # 視野角が，画像の大きさに効いてくる
    theta_fov = np.pi / 3
    f = 1

    # 透視画像の大きさを決定
    w = 2 * f * np.tan(theta_fov / 2)
    h = w

    # equi画像の横幅と縦幅の設定
    n_xe = 2 * hemisphere
    n_ye = hemisphere
    # print("hemisphere", n_ye)

    if base == (0, 0):
# if min(img_ori.shape) == 3:
# base = np.zeros((n_ye, n_xe, 3))
# else:
        # print("chose 1channel")
        base = np.zeros((n_ye, n_xe))
    else:
        base = base

    # 透視画像から，画素を球面上に持ってくる
    for i in range(n_x):
        for j in range(n_y):
            i_p, j_p = (i, j)

            x_c = (i_p - (n_x / 2)) * w / n_x
            y_c = (j_p - (n_y / 2)) * h / n_y
            z_c = f
            P = (x_c * np.sin(phi_v) + y_c * np.cos(theta_v) *
                 np.cos(phi_v) + z_c * np.sin(theta_v) * np.cos(phi_v),
                 -1 * x_c * np.cos(phi_v) + y_c * np.cos(theta_v) *
                 np.sin(phi_v) + z_c * np.sin(theta_v) * np.sin(phi_v),
                 -1 * y_c * np.sin(theta_v) + z_c * np.cos(theta_v))

            theta = np.arctan2(np.sqrt(P[0]**2 + P[1]**2),
                               P[2], dtype=np.float32)
            phi = np.arctan2(P[1], P[0], dtype=np.float32)
            phi += np.pi
            if np.isnan(phi):
                print("phi is nan")
                phi = 0
            if not 0 <= phi and phi <= 2*np.pi:
                print(phi)

            i_e = int(n_xe * (1 - phi / (2 * np.pi)))
            j_e = int(n_ye * theta / np.pi)
            # base[abs(j_e), abs(i_e)] = img_ori[j_p, i_p]
            # if base[j_e, i_e].all() == 0:
            if i_e == n_xe:
                i_e -= 1
            if j_e == n_ye:
                j_e -= 1
            base[j_e, i_e] = img_ori[j_p, i_p]

    base = np.asarray(base, dtype=np.uint8)
    # base = np.asarray(base, dtype=np.float32)
    # base = base[:, ::-1, :]
    return base


def pers2equi(input_image, start, hemisphere=200, r=1):  # {{{
    '''perspective imageをequirectangularに変更するequirectand
    input
    ------------
    hemisphere: 半球あたりのピクセル数
    r: 球の半径の値
    input_image: 入力となるperspective画像, numpy配列
    注意！np.asarrayとするときに、floatを指定してはいけない。
    start: どの位置から画像を始めるか[1,]

    return
    -------------
    array: equirectangular画像[0, 255]
    '''
    hemisphere = hemisphere
    r = r
    img = input_image
    channel = img.shape[-1]
    array = np.zeros((hemisphere, 2 * hemisphere, channel))

    theta = np.linspace(0, np.pi, hemisphere, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, 2 * hemisphere, endpoint=True)

    phi_v, theta_v = np.meshgrid(phi, theta, sparse=True)

    start = start
    end = img.shape[1]
    # print("IMG_SIZE: "+str(img.shape[1]))
    # print("start: "+str(start))
    for i in range(end):

        # ラジアンを計算して，倍率の計算
        rad = r * np.sin(theta_v[start + i])
        dis = 1 / rad

        # times = 2*hemisphere - 3 * i
        to_resize = np.int(dis * img.shape[1])
        if to_resize >= 2 * hemisphere:
            # print("over size at iteration "+str(i))
            to_resize = 2 * hemisphere
        left_pos = np.int((2 * hemisphere - to_resize) / 2)
        tmp = img[np.newaxis, i, :, :]
        if tmp.shape[-1] == 3:
            tmp = Image.fromarray(tmp, mode='RGB')
        elif tmp.shape[-1] == 1:
            tmp = Image.fromarray(tmp[:, :, 0], mode='L')
        tmp = tmp.resize((to_resize, 1), Image.BILINEAR)
        # opencvが使えない状況があるからPILで実装
        # tmp = cv.resize(tmp, (to_resize, 1), interpolation=cv.INTER_LINEAR)
        # print(tmp.shape)
        if channel == 1:
            array[start + i, left_pos:left_pos + to_resize, 0] = tmp
        else:
            array[start + i, left_pos:left_pos + to_resize, :] = tmp

    # array /= 255.
    # array = np.asarray(array, dtype=np.float32)
    # plt.figure()
    # plt.imshow(array, interpolation="none")
    # plt.show()
    return array  # }}}


def equi2sphere(pos, img_equi):  # {{{
    '''equirectangular画像を球面配列にする
    input
    ------------
    pos:球面配列の球座標が入ったpos配列(fish.make_considered_pictureで作成)
    img_equi:equi画像


    out
    ------------
    im_sphere: 画素値が入った球面配列
    '''
    array = img_equi
    # check
    _W, _H = img_equi.shape[:2]
    if not _H / _W == 2:
        print('invalid shape')
        return
    hemisphere = _W
    # print("hemisphere size: "+str(hemisphere))
    A, I, J = pos.shape[:3]
    im_sphere = np.zeros((3, A, I, J))
    # print("make sphere image with size: ", im_sphere.shape)
    for area in range(1, A):
        for i_axis in range(1, I - 1):
            for j_axis in range(1, J - 1):
                xe = (hemisphere - 1) * pos[area, i_axis, j_axis, 0] / np.pi
                ye = (2 * hemisphere - 1) * \
                    pos[area, i_axis, j_axis, 1] / (2 * np.pi)
                # ye = (hemisphere -1) * pos[area, i_axis, j_axis, 1] / np.pi
                xe = np.int(xe)
                ye = np.int(ye)
                try:
                    value = array[xe, ye]
                except:
                    ye = np.int(2*hemisphere) - ye
                    value = array[xe, ye]
                    # print(xe, ye)
                    # print(area, i_axis, j_axis)

                # value = fish.bilinear_interpolate(im=array, x=xe, y=ye)
                im_sphere[:, area, i_axis, j_axis] = value
    return im_sphere  # }}}


def make_sphere_img(input_image, start, pos, base=(0, 0),
                    hemisphere=200, r=1, return_equ=True):
    # array = pers2equi(input_image=input_image,
    # start=start, hemisphere=hemisphere, r=r)
    array = pers2equi2(input_image=input_image, start=start,
                       base=base,
                       hemisphere=hemisphere, r=r)
    if return_equ is True:
        return array
    else:
        img_out = equi2sphere(pos=pos, img_equi=array)
        return array, img_out


def make_sphere_batch(train_generator, pos, base=(0, 0), start=5,
                      hemisphere=400, r=3, return_equ=True):
    if min(train_generator.shape) == 3:
        print('loading cifar10')
        print(train_generator.shape)
        data = train_generator
        # channel = 3
    else:
        print('loading MNIST')
        print(train_generator.shape)
        data = train_generator
        # data = data.transpose(0, 2, 3, 1)
    data = np.asarray(data, dtype=np.uint8)
    equi_list = []
    sphere_list = []
    append_equi = equi_list.append
    append_sphere = sphere_list.append
    print('processing sphere image...')
    for i in tqdm(range(data.shape[0])):
        # if start == 'random':
        start = np.random.randint(3, 7)
        target = data[i]
        target = Image.fromarray(target)
        target = target.resize((100, 100))
        target = np.asarray(target, dtype=np.uint8)
        img_equi, img_sphere = make_sphere_img(
            input_image=target,
            start=start,
            pos=pos,
            base=base,
            hemisphere=hemisphere,
            r=r,
            return_equ=return_equ
        )
        append_equi(img_equi)
        append_sphere(img_sphere)
    im_equi = np.array(equi_list, dtype=np.float32)
    im_sphere = np.array(sphere_list, dtype=np.float32)

    return im_equi, im_sphere


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit("Usage: python image_tramsfer.py \
                 [img_path] [start] [level] [hemisphere]")

    input_img_path = sys.argv[1]
    start = sys.argv[2]
    level = sys.argv[3]
    hemisphere = sys.argv[4]

    img_ori = Image.open(input_img_path)
    img_ori = img_ori.resize((100, 100))
    img_ori = np.asarray(img_ori, dtype=np.uint8)
    img = img_ori
    # img = cv.imread(input_img_path)
    # img = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img_fish, pos = fish.make_considered_picture(
        img=img, level=level, return_array=1)

    make_sphere_img(img, start, pos, hemisphere)

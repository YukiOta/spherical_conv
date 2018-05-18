# coding: utf-8

from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from multiprocessing import Pool

from keras.datasets import mnist
# from keras.utils import np_utils

import numpy as np
import sys
import os
import argparse
sys.path.append('..')
import fish2hemisphere_local as fish
import image_transfer as it

seed = 100


def resize_image(array, size=(100, 100)):
    tmp = array
    tmp = Image.fromarray(np.uint8(tmp*255))
    tmp = tmp.resize(size, Image.BILINEAR)
    tmp = np.asarray(tmp, dtype=np.float32)
    tmp = tmp/255
    return tmp


def make_sphere_image_random(i, x, y, point):  # {{{
    target_ori = 0
    # point = make_point(n_clusters)
    img_equi = (0, 0)
    for k in range(len(point)):
        point_num = k

        # choose start and r
        start = point[point_num][1]
        r = point[point_num][0]

        # select image randomly
        np.random.seed(seed)
        num = np.random.randint(data_num)

        img_ori = x[num]
        target_ori += y[num]

        img_ori = resize_image(array=img_ori*255, size=(100, 100))
        if point_num == 0:
            img_equi = it.pers2equi2(img_ori*255, start=start,
                                     hemisphere=50, r=r)
        else:
            img_equi = it.pers2equi2(img_ori*255, start=start,
                                     hemisphere=50, r=r, base=img_equi)
    # append_x(img_equi)
    # append_y(target_ori)
    return [img_equi, target_ori]
# }}}


def make_sphere_image_unique(i, im_list, point, chosed_num, pos=None):
    mnist_list = []
    index_list = []
    for num_c in range(len(chosed_num)):
        tmp = im_list[chosed_num[num_c]]
        mnist_list.append(tmp)
        np.random.seed(seed)
        index_list.append(np.random.randint(len(tmp)))

    y_tmp = np.eye(10)[chosed_num]
    y_tmp = np.sum(y_tmp, axis=0)

    # point = make_point(n_clusters)
    img_equi = (0, 0)
    # print('point length', len(point))
    for k in range(len(point)):
        point_num = k

        # choose start and r
        start = point[point_num][1]
        r = point[point_num][0]

        img_ori = mnist_list[k][index_list[k]]

        img_ori = resize_image(array=img_ori*255, size=(100, 100))
        if point_num == 0:
            img_equi = it.pers2equi2(img_ori*255, start=start,
                                     hemisphere=50, r=r)
        else:
            img_equi = it.pers2equi2(img_ori*255, start=start,
                                     hemisphere=50, r=r, base=img_equi)
    if pos is not None:
        img_equi = img_equi.astype('float32')
        im_sphere = it.equi2sphere(pos, img_equi*255)
        imlist_original = []
        add_1 = imlist_original.append
        for i in range(1, 6):
            tmp = im_sphere[:, i, :, :]
            tmp = tmp.transpose(1, 2, 0)
            tmp = np.asarray(tmp, dtype=np.float32)
            to_show1 = tmp[1:-1, 1:-1, :]
            add_1(to_show1)

        tmp_ = imlist_original
        shape = tmp_[0].shape
        # base = np.zeros((int(shape[0]*5), int(shape[0]*6)))
        base = np.zeros((int(shape[0]*5), int(shape[0]*6), 3))

        for i in range(len(tmp_)):
            tmp = tmp_[-(i+1)][:, :, :]
            x_start = shape[0]*(i)
            x_end = shape[0]*(i+1)
            y_start = shape[0]*(4-i)
            y_end = shape[0]*(6-i)
            base[x_start:x_end, y_start:y_end] = tmp
        img_equi = base
    # append_x(img_equi)
    # append_y(target_ori)
    return img_equi, y_tmp


def wrapper_make_sphere_image(args):
    return make_sphere_image_unique(*args)


# {{{
def make_spherical_mnist_rondom(x, y):
    # list of spherical mmist
    n_clusters = 3

    global data_num
    data_num = len(x)
    print(data_num)
    tutumimono = [[i, x, y, make_point(n_clusters, i)] for i in tqdm(range(data_num))]
    p = Pool(20)
    output = p.map(wrapper_make_sphere_image, tutumimono)
    p.close()

    # for i in tqdm(range(data_num)):
    # x = np.array(spherical_im)
    # y = np.array(spherical_target)
    return output
# }}}


def make_spherical_mnist(x, y, pos=None):
    # list of spherical mmist
    '''
    pos None is equi
    pos not None is picapica
    '''
    n_clusters = 7

    im_list = []
    for i in range(10):
        num = i
        index = np.where(y == num)
        im_list.append(x[index])

    print('the number of data:', len(x))
    numbers = [i for i in range(10)]

    if pos is None:
        np.random.seed(seed)
        tutumimono = [[i, im_list, make_point(n_clusters, i),
                       np.random.choice(numbers, n_clusters, replace=False)] for i in tqdm(range(len(x)))]
    else:
        print("add pos")
        np.random.seed(seed)
        tutumimono = [[i, im_list, make_point(n_clusters, i),
           ?!?jedi=0,             np.random.choice(numbers, n_clusters, replace=False), pos] for i in tqdm(range(len(x)))]?!? (*_*param processes=None*_*, param initializer=None, param initargs=(), param maxtasksperchild=None) ?!?jedi?!?
    p = Pool(2)
    output = p.map(wrapper_make_sphere_image, tutumimono)
    p.close()

    # for i in tqdm(range(data_num)):
    # x = np.array(spherical_im)
    # y = np.array(spherical_target)
    return output


def make_point(n_clusters, i):  # {{{
    # find initial random position
    # make theta phi grid
    phi = np.linspace(0, np.pi*2, num=200, retstep=False)
    theta = np.linspace(0, np.pi, num=100, retstep=False)
    XX, YY = np.meshgrid(phi, theta)
    XX = XX.reshape(-1)
    YY = YY.reshape(-1)
    XX = XX[:, np.newaxis]
    YY = YY[:, np.newaxis]
    mesh = np.concatenate((XX, YY), axis=1)
    # calculate init
    n_clusters = n_clusters
    dist = KMeans(n_clusters=n_clusters, random_state=i,
                  n_init=1, max_iter=1).fit(mesh)
    point = dist.cluster_centers_
    # calculate parameter (0:r, 1:start)
    point /= np.array([np.pi, np.pi])
    return point  # }}}


def rotation_xyz(pos, alpha, mode):
    np.seterr(divide='ignore')
    pos_new1 = np.zeros(pos.shape)

    theta = pos[1:, :, :, 0]
    phi = pos[1:, :, :, 1]

    r = 1
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x = x[np.newaxis, :, :, :]
    y = y[np.newaxis, :, :, :]
    z = z[np.newaxis, :, :, :]
    xyz = np.concatenate((x, y, z), axis=0)

    rot_x = np.array([[1, 0, 0],
                               [0, np.cos(alpha), np.sin(alpha)],
                               [0, -1*np.sin(alpha), np.cos(alpha)]])
    rot_y = np.array([[np.cos(alpha), 0, -1*np.sin(alpha)],
                               [0, 1, 0],
                               [np.sin(alpha), 0, np.cos(alpha)]])
    rot_z = np.array([[np.cos(alpha), np.sin(alpha), 0],
                               [-1*np.sin(alpha), np.cos(alpha), 0],
                               [0, 0, 1]])
    shape = xyz.shape
    xyz = xyz.transpose(1, 2, 3, 0)

    if mode == "x":
        print("mode: ", mode)
        xyz_new = np.dot(xyz, rot_x)
    elif mode == "z":
        print("mode: ", mode)
        xyz_new = np.dot(xyz, rot_z)
    elif mode == "y":
        print("mode: ", mode)
        xyz_new = np.dot(xyz, rot_z)
    elif mode == "xz":
        print("mode: ", mode)
        xyz_new = np.dot(xyz, rot_x)
        xyz_new = np.dot(xyz_new, rot_z)
    elif mode == "xy":
        print("mode: ", mode)
        xyz_new = np.dot(xyz, rot_x)
        xyz_new = np.dot(xyz_new, rot_y)
    else:
        print("invalid mode")
        return
    xyz_new = xyz_new.transpose(3, 0, 1, 2)

    pos_new1[1:, :, :, 0] = np.arctan2(np.sqrt(xyz_new[0]**2 + xyz_new[1]**2), xyz_new[2])
    pos_new1[1:, :, :, 1] = np.arctan2(xyz_new[1], xyz_new[0])
    # pos_new1[:, :, :, 1] += np.pi / 2
    return pos_new1


def main(args):
    data_dir, save_dir = args.data_dir, args.save_dir

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # y_test = np_utils.to_categorical(y_test)
    # y_train = np_utils.to_categorical(y_train)

    img_ori = np.zeros((224, 224, 3))
    level = 4
    img_fish, pos = fish.make_considered_picture(img=img_ori, level=level, return_array=1)
    pos_x = rotation_xyz(pos, alpha=np.pi/2, mode="x")
    print("level: ", level)
    print(img_fish.shape)

    print('making train samples')
    # x_train, y_train = make_spherical_mnist(x=x_train[:10000], y=y_train[:10000])
    # output_train = make_spherical_mnist(x=x_train[:30000], y=y_train[:30000])
    output_train = make_spherical_mnist(x=x_train[:10000], y=y_train[:10000], pos=pos_x)
    print("arraying")
    output_train = np.array(output_train)
    print('making test samples')
    # x_test, y_test = make_spherical_mnist(x=x_test[:1000], y=y_test[:1000])
    output_test = make_spherical_mnist(x=x_test, y=y_test, pos=pos_x)
    output_test = np.array(output_test)

    title = 'sphere_mnist_pica_level4_cluster7x_10000.npz'
    print(title)
    path_sp = os.path.join(save_dir, title)
    # np.save(path_sp, output)
    np.savez(path_sp,
             train=output_train,
             test=output_test,
             )

    pos_xz = rotation_xyz(pos, alpha=np.pi/2, mode="xz")
    print('making train samples')
    output_train = make_spherical_mnist(x=x_train[:10000], y=y_train[:10000], pos=pos_xz)
    print("arraying")
    output_train = np.array(output_train)
    print('making test samples')
    output_test = make_spherical_mnist(x=x_test, y=y_test, pos=pos_xz)
    output_test = np.array(output_test)

    title = 'sphere_mnist_pica_level4_cluster7xz_10000.npz'
    # title =  'sphere_mnist_unique_v1_cluster5_10000.npz'
    print(title)
    path_sp = os.path.join(save_dir, title)
    # np.save(path_sp, output)
    np.savez(path_sp,
             train=output_train,
             test=output_test,
             )


    print('making train samples')
    output_train = make_spherical_mnist(x=x_train[:10000], y=y_train[:10000])
    print("arraying")
    output_train = np.array(output_train)
    print('making test samples')
    output_test = make_spherical_mnist(x=x_test, y=y_test)
    output_test = np.array(output_test)

    title =  'sphere_mnist_unique_v1_cluster7_10000.npz'
    print(title)
    path_sp = os.path.join(save_dir, title)
    # np.save(path_sp, output)
    np.savez(path_sp,
             train=output_train,
             test=output_test,
             )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_dir',
        default='../data/',
        help='choose your data dir'
    )
    parser.add_argument(
        '-s', '--save_dir',
        default='../data/',
        help='choose your save dir'
    )

    args = parser.parse_args()
    data_dir, save_dir = args.data_dir, args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if not os.path.isdir(data_dir):
        print('make data_dir', data_dir)
        os.makedirs(data_dir)

    main(args)

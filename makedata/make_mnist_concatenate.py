# coding: utf-8

from tqdm import tqdm
from multiprocessing import Pool

from keras.datasets import mnist
# from keras.utils import np_utils

import numpy as np
import sys
import os
import argparse
sys.path.append('..')
# import fish2hemisphere_local as fish


def make_image(i, im_list, chosed_num):
        im_1 = im_list[chosed_num[0]]
        im_2 = im_list[chosed_num[1]]
        im_3 = im_list[chosed_num[2]]
        index_1 = np.random.randint(len(im_1))
        index_2 = np.random.randint(len(im_2))
        index_3 = np.random.randint(len(im_3))

        x_tmp = np.concatenate((im_1[index_1],
                                im_2[index_2], im_3[index_3]), axis=1)
        y_tmp = np.eye(10)[chosed_num]
        y_tmp = np.sum(y_tmp, axis=0)
        return x_tmp, y_tmp


def wrapper_make_image(args):
    return make_image(*args)


def make_3_mnist(x, y):
    im_list = []

    # separate data with reagard to NUMBER
    for i in range(10):
        num = i
        index = np.where(y == num)
        im_list.append(x[index])

    # 重複のない様に選ぶ
    numbers = [i for i in range(10)]
    # x_new = []
    # y_new = []
    # appendx = x_new.append
    # appendy = y_new.append
    tutumimono = [[i, im_list, np.random.choice(numbers, 3, replace=False)] for i in tqdm(range(len(x)))]
    p = Pool(10)
    output = p.map(wrapper_make_image, tutumimono)
    p.close()

    return output


def main(args):
    data_dir, save_dir = args.data_dir, args.save_dir
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('making train samples')
    output_train = make_3_mnist(x=x_train, y=y_train)
    print("arraying")
    output_train = np.array(output_train)
    print('making test samples')
    output_test = make_3_mnist(x=x_test, y=y_test)
    output_test = np.array(output_test)

    path_sp = os.path.join(save_dir, 'concatenate3_mnist_v1.npz')
    # np.save(path_sp, output)
    np.savez(path_sp,
             train=output_train,
             test=output_test
             )
    print('done')


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

# coding: utf-8

from sphere_convnet import SphereConv
from trainer import Trainer
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import fish2hemisphere_local as fish
import image_transfer as it
import argparse
import os
import time
import datetime as dt
# import seaborn as sns


def make_data(array):
    x_tmp = []
    y_tmp = []
    appendx = x_tmp.append
    appendy = y_tmp.append

    for i in range(len(array)):
        appendx(array[i][0])
        appendy(array[i][1])

    x_tmp = np.array(x_tmp)
    y_tmp = np.array(y_tmp)
    return x_tmp, y_tmp


def wrapper(args):
    return _get_array(*args)


def get_sphere_array(x, pos):
    p = Pool(20)
    tutumimono = [[i, x, pos] for i in tqdm(range(len(x)))]
    output = p.map(wrapper, tutumimono)
    p.close()
    output = np.array(output)
    return output


def _get_array(i, x, pos):
    tmp = x[i]*255
    im_sphere = it.equi2sphere(pos, tmp)
    return im_sphere


def main(args):
    data_dir, save_dir, mode = args.data_dir, args.save_dir, args.mode

    # pos
    img_ori = np.zeros((224, 224, 3))
    level = 4
    img_fish, pos = fish.make_considered_picture(img=img_ori, level=level, return_array=1)
    print("level: ", level)
    print(img_fish.shape)

    # data load
    ndar = np.load(data_dir)
    train = ndar['train']
    test = ndar['test']

    x_train, y_train = make_data(train)
    x_test, y_test = make_data(test)

    x_train = get_sphere_array(x_train, pos)
    x_test = get_sphere_array(x_test, pos)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    if x_train.shape[-1] == 3:
        print('change channel')
        x_train = x_train[:, :, :, 0]
        x_test = x_test[:, :, :, 0]
    # if len(x_test) > 1000:
    #     tmp = 1000
    #     x_train = np.concatenate((x_train, x_test[tmp:]), axis=0)
    #     x_test = x_test[:tmp]
    #     y_train = np.concatenate((y_train, y_test[tmp:]), axis=0)
    #     y_test = y_test[:tmp]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    # パラメータの設定
    hemisphere = 50
    Q = 2**level
    title = 'SphereConv'
    batch_size = 100
    max_epochs = 10
    out_num = 10
    print('conducting in SphereConv')
    # パラメータの吐き出し
    with open(save_dir + 'setting.txt', 'w') as f:
        f.write('save_dir: ' + save_dir + '\n')
        f.write('hemisphere: ' + str(hemisphere) + '\n')
        f.write('level: ' + str(level) + '\n')
        f.write('Q: ' + str(Q) + '\n')
        f.write('batch_size: ' + str(batch_size) + '\n')
        f.write('max_epochs: ' + str(max_epochs) + '\n')
        f.write('out_num :' + str(out_num) + '\n')
        f.write('train : 10000' + '\n')
        f.write('test 1000 :' + '\n')
        f.write('COMMENT: SGD, SphereConv\n')

    print('train size;', x_train.shape)
    network = SphereConv(input_dim=(3, 6, Q+2, 2*Q+2),
                         conv_param={'filter_num': 32,
                                     'filter_size': 7,
                                     'pad': 0, 'stride': 1},
                         hidden_size=128, output_size=out_num,
                         weight_init_std='he', level=level)

    trainer = Trainer(network, x_train, y_train, x_test[:100], y_test[:100],
                      epochs=max_epochs, mini_batch_size=batch_size,
                      optimizer='SGD', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=100)
    trainer.train()

    # パラメータの保存
    network.save_params(save_dir+"params.pkl")
    print("Saved Network Parameters!")

    try:
        print('save acc')
        a = np.array(trainer.train_acc_list)
        b = np.array(train_acc_list.test_acc_list)
        np.savez('acc.npz', train=a, test=b)
    except:
        print('error in saving acc')
    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.savefig(save_dir+"test.png")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-D', '--data_dir',
        default='../data/',
        help='choose your data dir'
    )
    parser.add_argument(
        '-S', '--save_dir',
        default='../data/',
        help='choose your data dir'
    )
    parser.add_argument(
        '-M', '--mode',
        default='main',
        help='choose your data dir'
    )
    today_time = dt.datetime.today().strftime('%Y_%m_%d')

    args = parser.parse_args()
    data_dir, save_dir, mode = args.data_dir, args.save_dir, args.mode
    save_dir = save_dir + today_time + '/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    start_t = time.time()
    main(args)
    elapsed_time = time.time() - start_t
    print('elapsed_time: {0}'.format(elapsed_time) + '[sec]')

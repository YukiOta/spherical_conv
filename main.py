# -*- coding: utf-8 -*-

# library
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import datetime as dt
from tqdm import tqdm

# pathの追加
sys.path.append('./utils')
sys.path.append('./conv')
import fish2hemisphere_local as fish
import image_transfer as it
from sphere_convnet import SphereConv
from trainer import Trainer
from make_data import make_data


'''並列処理'''  # {{{
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
# }}}


def main(args):

    #######################
    # ディレクトリパスの読み込み
    #######################
    # 今日の日付
    today_time = dt.datetime.today().strftime("%Y_%m_%d")

    # 各ディレクトリの設定
    DATA_DIR, SAVE_DIR = \
        args.data_dir, args.save_dir

    # 'SAVE_DIR'に今日の日付を足す
    SAVE_DIR = os.path.join(SAVE_DIR, today_time)

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    #######################
    # Hyper Parameter
    #######################
    param_dic = {}
    level = param_dic['level'] = 4
    hemisphere = param_dic['hemisphere'] = 50
    Q = param_dic['Q'] = 2**level
    title = param_dic['title'] = 'SphereConv'

    batch_size = param_dic['batch_size'] = 100
    max_epochs = param_dic['max_epochs'] = 10
    evaluate_num = param_dic['evaluate_sample_num_per_epoch'] = 100
    out_num = param_dic['out_num'] = 10
    filter_num = param_dic['filter_num'] = 32
    filter_size = param_dic['filter_size'] = 7
    padding = param_dic['padding'] = 0
    stride = param_dic['stride'] = 1
    hidden_size = param_dic['hidden_size'] = 128
    optimizer = param_dic['optimizer'] = 'SGD'
    lr = param_dic['lr'] = 0.001
    weight_init_std = param_dic['weight_init_std'] = 'he'

    # 保存
    with open(SAVE_DIR + 'setting.txt', 'a') as f:
        for key, value in param_dic.items():
            f.write(str(key) + ':' + str(value))
            f.write('\n')

    #######################
    # 球面座標の獲得
    #######################
    img_ori = np.zeros((224, 224, 3))
    img_fish, pos = fish.make_considered_picture(img=img_ori, level=level, return_array=1)
    # print("level: ", level)
    # print(img_fish.shape)

    #######################
    # Load Data
    #######################
    ndar = np.load(data_dir)
    train = ndar['train']
    test = ndar['test']

    #######################
    # Make data
    # データをネットワークに適した形に変形する
    #######################
    x_train, y_train = make_data(train)
    x_test, y_test = make_data(test)

    #######################
    # 球面画像の獲得
    #######################
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

    #######################
    # Train
    #######################
    network = SphereConv(input_dim=(3, 6, Q+2, 2*Q+2),
                         conv_param={'filter_num': filter_num,
                                     'filter_size': filter_size,
                                     'pad': padding, 'stride': stride},
                         hidden_size=hidden_size, output_size=out_num,
                         weight_init_std=weight_init_std, level=level)

    trainer = Trainer(network, x_train, y_train, x_test[:100], y_test[:100],
                      epochs=max_epochs, mini_batch_size=batch_size,
                      optimizer=optimizer, optimizer_param={'lr': lr},
                      evaluate_sample_num_per_epoch=evaluate_num)
    trainer.train()

    #######################
    # パラメータの保存
    #######################
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
    '''
    --data_dir : 入力データがあるディレクトリパス
    --save_dir : 結果を保存するディレクトリのパス
    '''
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
    today_time = dt.datetime.today().strftime('%Y_%m_%d')
    args = parser.parse_args()

    #######################
    # 関数の実行
    #######################
    start_t = time.time()
    main(args)
    elapsed_time = time.time() - start_t
    print('Elapsed Time: {0}'.format(elapsed_time) + '[sec]')

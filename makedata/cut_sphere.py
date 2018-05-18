# coding: utf-8
import numpy as np
import os
import sys
sys.path.append('..')
import fish2hemisphere_local as fish
import image_transfer as it
# from keras.datasets import mnist, cifar10, cifar100
# from keras.utils import np_utils
from tqdm import tqdm
from PIL import Image


(x_train, t_train), (x_test, t_test) = cifar100.load_data()
# load sphere array
# x_train, t_train = x_train[:50], t_train[:50000]
# x_test, t_test = x_test[:10], t_test[:10000]

level = 6
hemisphere = 60
r = 1
Q = 2**level
img_ori = np.zeros((224, 224, 3))
img_fish, pos = fish.make_considered_picture(
        img=img_ori, level=level, return_array=1)

# process equi image
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_train_sp = it.make_sphere_batch(
        train_generator = x_train,
        pos=pos,
        start='random',
        hemisphere=hemisphere,
        r=r,
        return_equ=False)
x_test, x_test_sp = it.make_sphere_batch(
        train_generator = x_test,
        pos=pos,
        start='random',
        hemisphere=hemisphere,
        r=r,
        return_equ=False)
x_train /= 255
x_test /= 255
x_train_sp /= 255
x_test_sp /= 255
# make target one-hot vector
t_test = np_utils.to_categorical(t_test)
t_train = np_utils.to_categorical(t_train)

input_shape = (hemisphere, 2*hemisphere, 3)
x_train = x_train.reshape(x_train.shape[0], hemisphere, 2*hemisphere, 3)
x_test = x_test.reshape(x_test.shape[0], hemisphere, 2*hemisphere, 3)

# 配列の保存
data_dir = '../../data/cifar100/'
path = os.path.join(data_dir, 'cifar100_startrandom_equi_v2.npz')
path_sp = os.path.join(data_dir, 'cifar100_startrandom_sphere_v2.npz')
# np.savez(path,
#         x_test=x_test,
#         x_train=x_train,
#         t_train=t_train,
#         t_test=t_test)
np.savez(path_sp,
        x_test=x_test_sp,
        x_train=x_train_sp,
        t_train=t_train,
        t_test=t_test)

'''
# 保存した、cifar10_sphereの読み込み
mnist_path = '../../data/cifar10/cifar10_startrandom_sphere_v2.npz'
ndarr = np.load(mnist_path)
x_train = x_train_sp  # ndarr['x_train']
x_test = x_test_sp  # ndarr['x_test']
# t_train = ndarr['t_train']
# t_test = ndarr['t_test']
# x_train = x_train_sp
# x_test = x_test_sp
x_train = x_train[:, :, 1:, 1:-1, 1:-1]
x_test = x_test[:, :, 1:, 1:-1, 1:-1]

print('processing train')
x_train = x_train[:, :, :, :, :]
print('processing train.')
x_train = x_train.transpose(0, 2, 3, 4, 1)
print('processing train..')
x_train = x_train[:, ::-1, :, :, :]
print('processing train...')
x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[3], x_train.shape[4])
print('processing train....')
# x_train = x_train[:, 100:200, :, :]
x_train = x_train[:, :, :, :]

print('processing test')
x_test = x_test[:, :, :, :, :]
print('processing test.')
x_test = x_test.transpose(0, 2, 3, 4, 1)
print('processing test..')
x_test = x_test[:, ::-1, :, :, :]
print('processing test...')
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[3], x_test.shape[4])
print('processing test....')
# x_test = x_test[:, 100:200, :, :]
x_test = x_test[:, :, :, :]
'''

mnist_path = '../../data/cifar100/cifar100_startrandom_sphere_v3.npz'
ndarr = np.load(mnist_path)
x_train = ndarr['x_train']
x_test = ndarr['x_test']
t_train = ndarr['t_train']
t_test = ndarr['t_test']

# reshape
x_train_ls = []
x_test_ls = []
append1 = x_train_ls.append
append2 = x_test_ls.append
for i in tqdm(range(len(x_train))):
    tmp = x_train[i]
    # H = tmp.shape[1] * 60 // tmp.shape[0]
    # W = 60
    # H = tmp.shape[1]//2
    # W = tmp.shape[0]//2
    # resize image here
    # tmp = Image.fromarray(np.uint8(tmp*255))
    # tmp = tmp.resize((H, W), Image.BILINEAR)
    tmp = np.asarray(tmp, dtype=np.float32)
    # tmp = tmp/255
    tmp = tmp[:, :-28, :]
    tmp = np.rot90(tmp, k=3)
    append1(tmp)

for i in tqdm(range(len(x_test))):
    tmp = x_test[i]
    # resize image here
    tmp = np.asarray(tmp, dtype=np.float32)
    tmp = tmp[:, :-28, :]
    tmp = np.rot90(tmp, k=3)
    append2(tmp)
x_train = np.array(x_train_ls)
x_test = np.array(x_test_ls)

print('saving')
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
print('saving.')
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
print(x_train.shape)

# 配列の保存
print('saving..')
data_dir = '../../data/cifar100/'
title = 'cifar100_startrandom_sphere_v5_resized.npz'
path_sp = os.path.join(data_dir, title)
np.savez(path_sp,
     x_train=x_train,
     x_test=x_test,
     t_train=t_train,
     t_test=t_test)

with open(data_dir + title +'_setting.txt', 'w') as f:
    f.write('image size: ' + str(x_train.shape[1:]) + '\n')
    f.write('training number' + str(x_train.shape[0]) + '\n')
    f.write('test number' + str(x_test.shape[0]) + '\n')
print('done')

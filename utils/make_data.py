# -*- coding: utf-8 -*-

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

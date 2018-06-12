# encoding: utf-8
""" 时序预测 工具函数
"""

import numpy as np


# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def create_dataset(dataset, input_size=1):
    data_X, data_y = [], []
    for i in range(input_size, dataset.shape[0]): # [)
        data_X.append(dataset[i-input_size: i, 0])
        data_y.append(dataset[i, 0])
    data_X, data_y = np.array(data_X), np.array(data_y)
    return data_X, data_y



if __name__ == '__main__':

    train_dataset = np.arange(0, 10).reshape((-1, 1))
    train_x, train_y = create_dataset(train_dataset, input_size=5)
    '''
        raw data  ->  train_x:
        [[0]          [[0 1 2 3 4]
         [1]           [1 2 3 4 5]
         [2]           [2 3 4 5 6]
         [3]           [3 4 5 6 7]
         [4]           [4 5 6 7 8]]
         [5]
         [6]          train_y:
         [7]          [5 6 7 8 9]
         [8]
         [9]]
    '''

    
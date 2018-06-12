# encoding: utf-8
""" 预测股票价格

@version 1.0.2 build 20180613
"""

from __future__ import print_function
import os
import argparse
import torch
from torch.autograd import Variable

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt

from models import RNN

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


parser = argparse.ArgumentParser(description='Pytorch Time Sequence Perdicton')
parser.add_argument('--input', default='input/', help='path to dataset')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--output', default='saves/', help='folder to output images and model checkpoints')
parser.add_argument('--epochs', type=int, default=8, metavar='EPOCHS', help='number of epochs to train (default: 8)')

args = parser.parse_args()



# Hyper Parameters
# HIDDEN_SIZE 调整为100、200，或 NUM_LAYERS 调整为 2，不明白为什么始终无法拟合❔😓
TIME_STEP = 1
INPUT_SIZE = 60
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1


# 加载训练好的模型
rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

rnn_state_dict = torch.load('%s/time_series_rnn_model_params.pkl' % args.output)
rnn.load_state_dict(rnn_state_dict)



# 循环预测接下来的数据
''' sample 示例
 1 2 3 4 5 6 7 8 9 10   --- predict ---> 11 12 13 14 15
 train[-5:]
         |          |
          --inputs--
'''
# 加载预处理过（正则化）的训练数据, 取最后的时间序列预测未来20几天
train_dataset = np.load('input/000001.XSHE_train.npy')
scaler = joblib.load('encoder/min_max_scaler.close.pkl')

# 循环预测接下来的数据
''' sample 示例
 1 2 3 4 5 6 7 8 9 10   --- predict ---> 11 12 13 14 15
 train[-5:]
         |          |
          --inputs--
'''
test_dataset = train_dataset[-INPUT_SIZE:]


pred_list = []
for i in range(22):  # 预测未来22天的数据
    # Reshaping (batch, time_step, input_size)
    inputs = test_dataset.reshape(-1, 1, 60)
    inputs = Variable(torch.from_numpy(inputs).float())
    
    hidden_state = None
    output, hidden_state = rnn(inputs, hidden_state)
    predicted = output.data.numpy()
    ''' [[ 0.88910466]] '''
    
    pred_list.append(predicted[0][0])

    # 加入新预测的价格，去掉顶部之前的价格
    test_dataset = np.concatenate((test_dataset, predicted), axis=0)
    test_dataset = test_dataset[1:]

# 预测的值是[0,1]，将该值转换回原始值
pred_list = np.reshape(pred_list, (-1, 1))
pred_list = scaler.inverse_transform(pred_list)




plt.figure(1, figsize=(12, 5))
# 绘制训练的股价
plt.plot(scaler.inverse_transform(train_dataset), 'b-', label='train')

# 绘制未来股价的真实价格
test_dataset = pd.read_csv('data/000001.XSHE_test.csv')
real_stock_price = test_dataset.iloc[:, 1:2].values
plt.plot(np.arange(len(train_dataset), len(train_dataset)+len(real_stock_price)), real_stock_price.flatten(), 'g-', label='real')

# 可视化预测未来几天的价格
plt.plot(np.arange(len(train_dataset), len(train_dataset)+len(real_stock_price)), pred_list, 'r:', label='predict')

plt.legend(loc='best')
plt.title('平安银行')
plt.show()




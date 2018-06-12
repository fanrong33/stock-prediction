# encoding: utf-8
""" 预测股票价格

@version 1.0.2 build 20180528
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

import torch
import torch.nn as nn
from torch.autograd import Variable

from models import RNN
import util

torch.manual_seed(1)


parser = argparse.ArgumentParser(description='Pytorch Time Sequence Perdicton')
parser.add_argument('--input', default='input/', help='path to dataset')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--output', default='saves/', help='folder to output images and model checkpoints')
parser.add_argument('--epochs', type=int, default=8, metavar='EPOCHS', help='number of epochs to train (default: 8)')

args = parser.parse_args()
CUDA = torch.cuda.is_available()

try:
    os.makedirs(args.output)
except OSError:
    pass



# Part 1 - Data Preprocessing

# Hyper Parameters
# HIDDEN_SIZE 调整为100、200，或 NUM_LAYERS 调整为 2，不明白为什么始终无法拟合❔😓
TIME_STEP = 1
INPUT_SIZE = 60
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1


# 加载预处理过（正则化）的训练数据
training_set_scaled = np.load('input/000001.XSHE_train.npy')
scaler = joblib.load('encoder/min_max_scaler.close.pkl')

# Creating a data structure with 60 timesteps and 1 output
train_X, train_y = util.create_dataset(training_set_scaled, input_size=INPUT_SIZE)
# print(train_X.shape)
''' (672, 60) '''

# Reshape为 (batch, time_step, input_size), 这是放入LSTM的shape
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
# print(train_X.shape)
''' (672, 1, 60) '''



# Part 2 - Building the RNN
rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)


optimiser = torch.optim.Adam(rnn.parameters(), lr=args.lr)
loss_func = nn.MSELoss()
if CUDA:
    loss_func.cuda()


plt.figure(1, figsize=(12, 5))
plt.ion()

hidden_state = None
for epoch in range(args.epochs):

    inputs = Variable(torch.from_numpy(train_X).float())
    labels = Variable(torch.from_numpy(train_y).float())
    if CUDA:
        inputs, labels = inputs.cuda(), labels.cuda()

    output, hidden_state = rnn(inputs, hidden_state)
    hidden_state = Variable(hidden_state.data)

    loss = loss_func(output.view(-1), labels)
    optimiser.zero_grad()
    loss.backward(retain_graph=True)           # back propagation !important
    optimiser.step()                           # update the parameters

    
    print('Epoch {}, Training Loss {}'.format(epoch,loss.data[0]))


    # plot
    plt.cla()
    plt.plot(scaler.inverse_transform(train_y.reshape(-1, 1)), 'b-', label='train')
    plt.plot(scaler.inverse_transform(output.data.numpy()), 'r-', label='fit')


    # 打印未来数据的真实股票价格
    dataset_test = pd.read_csv('data/000001.XSHE_test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    plt.plot(np.arange(len(train_y), len(train_y)+len(real_stock_price)), real_stock_price.flatten(), 'g-')


    ''' sample 示例
     1 2 3 4 5 6 7 8 9 10   --- predict ---> 11 12 13 14 15
     train[-5:]
             |          |
              --inputs--
    '''
    test_dataset = training_set_scaled[-INPUT_SIZE:]

    pred_list = []
    for i in range(len(real_stock_price)):
        # Reshaping (batch, time_step, input_size)
        test_inputs = test_dataset.reshape(-1, 1, 60)
        test_inputs = Variable(torch.from_numpy(test_inputs).float())
        
        hidden_state2 = None
        output, hidden_state2 = rnn(test_inputs, hidden_state2)
        predict_price = output.data.numpy()
        pred_list.append(predict_price[0][0])

        # 加入新预测的价格，去掉顶部之前的价格
        test_dataset = np.concatenate((test_dataset, predict_price), axis=0)
        test_dataset = test_dataset[1:]
    
    # 预测的值是[0,1]，将该值转换回原始值
    pred_list = np.reshape(pred_list, (-1, 1))
    pred_list = scaler.inverse_transform(pred_list)
    
    # 可视化预测未来几天的价格
    plt.plot(np.arange(len(train_y), len(train_y)+len(real_stock_price)), pred_list, 'y:')

    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.05)

 # Do checkpointing
torch.save(rnn.state_dict(), '%s/time_series_rnn_model_params.pkl' % args.output)  # 只保存网络中的参数（速度快，占内存少）


plt.ioff()
plt.show()




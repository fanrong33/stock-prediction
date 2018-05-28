# encoding: utf-8
""" 预测股票价格

@version 1.0.2 build 20180528
"""


# Part 1 - Data Preprocessing

import numpy as np
import pandas as pd


# Hyper Parameters
# HIDDEN_SIZE 调整为100、200，或 NUM_LAYERS 调整为 2，不明白为什么始终无法拟合❔😓
TIME_STEP = 1
INPUT_SIZE = 60
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1
LR = 0.001


# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def create_dataset(dataset, input_size=1):
    data_X, data_y = [], []
    for i in range(input_size, dataset.shape[0]): # [)
        data_X.append(dataset[i-input_size: i, 0])
        data_y.append(dataset[i, 0])
    data_X, data_y = np.array(data_X), np.array(data_y)
    return data_X, data_y

train_dataset = np.arange(0, 10).reshape((-1, 1))
# print(train_dataset)
'''
[[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]
'''
train_x, train_y = create_dataset(train_dataset, input_size=5)
# print(train_x)
'''
[[0 1 2 3 4]
 [1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]]
'''
# print(train_y)
'''
[5 6 7 8 9]
'''


# Importing the training set
dataset_train = pd.read_csv('000001.XSHE_train.csv')
training_set = dataset_train.iloc[:, 1:2].values
'''
[[ 13.26]
 [ 13.58]
 [ 13.28]
 [ 13.21]]
'''
# print(training_set.shape)
''' (732, 1) '''

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output
train_X, train_y = create_dataset(training_set_scaled, input_size=INPUT_SIZE)
# print(train_X.shape)
''' (672, 60) '''


# Reshape为 (batch, time_step, input_size), 这是放入LSTM的shape
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
# print(train_X.shape)
''' (672, 1, 60) '''


import torch.nn as nn
import torch
from torch.autograd import Variable

torch.manual_seed(1)


# Part 2 - Building the RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h_state):

        r_out, h_state = self.rnn(x, h_state)
        
        # hidden_size = h_state[-1].size(-1)

        r_out = r_out.view(-1, self.hidden_size)
        outs = self.fc(r_out)
        return outs, h_state



rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)


optimiser = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

hidden_state = None

import matplotlib.pyplot as plt

plt.figure(1, figsize=(12, 5))
plt.ion()

for epoch in range(1000):

    inputs = Variable(torch.from_numpy(train_X).float())
    labels = Variable(torch.from_numpy(train_y).float())
    

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

    # Getting the real stock price of feture test data
    dataset_test = pd.read_csv('000001.XSHE_test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    plt.plot(np.arange(len(train_y),len(train_y)+len(real_stock_price)), real_stock_price.flatten(), 'g-')

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
    plt.plot(np.arange(len(train_y),len(train_y)+len(real_stock_price)), pred_list, 'y:')

    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.05)


plt.ioff()
plt.show()



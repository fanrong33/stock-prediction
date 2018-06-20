# encoding: utf-8
""" 数据预处理
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


# Importing the training set
dataset_train = pd.read_csv('data/000001.XSHE_train.csv')
training_set = dataset_train.iloc[:, 1:2].values
print(training_set)
'''
[[ 13.26]
 [ 13.58]
 [ 13.28]
 [ 13.21]]
'''
# print(training_set.shape)
''' (732, 1) '''


# Feature Scaling
scaler = StandardScaler()
training_set_scaled = scaler.fit_transform(training_set)
print(training_set_scaled)

# 保存预处理后的数据为input/x.npy
np.save('input/000001.XSHE_train', training_set_scaled)

# 持久化encoder对象, 以便预测的时候使用
joblib.dump(scaler, 'encoder/standard_scaler.close.pkl')




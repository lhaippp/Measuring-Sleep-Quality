# -*- coding: utf-8 -*-
"""
Created on 07/03/2019
@author: Haipeng 
"""
import pandas as pd 
import numpy as np
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn import metrics
from tf_utils import random_mini_batches,convert_to_one_hot
import lightgbm as lgb

buffer_df = pd.read_csv('SleepQuality_After_Cleaning.csv')

'''Aggregation from 5 to 3:
    1-2: 0 (Bad)
    3:   1 (Mean)
    4-5: 2 (Good)
'''

buffer_df['sleep_quality'] = (buffer_df['ms204a']+buffer_df['ms204b']+buffer_df['ms204c'])/3
buffer_df.sleep_quality[buffer_df['sleep_quality'] < 3] = 0
buffer_df.sleep_quality[buffer_df['sleep_quality'] == 3] = 1
buffer_df.sleep_quality[buffer_df['sleep_quality'] > 3] = 2



trainSet_X = buffer_df[['avg23bpd_s2','avg23bps_s2','ai_all','rdi0p', 'nsupinep', 'pctlt75', 'pctlt80', 'pctlt85', 'pctlt90',
                     'slp_eff', 'slp_lat', 'slpprdp', 'supinep', 'times34p', 'timest1p', 'timest2p', 'waso']]

# trainSet_y = buffer_df.loc[:,['ms204a', 'ms204b', 'ms204c']]
# Sleep Quality from light to dark: 1 -> 5
# trainSet_y = buffer_df.loc[:,['ms204a']]
# trainSet_y = buffer_df.loc[:,['ms204b']]
# trainSet_y = buffer_df.loc[:,['ms204c']]
trainSet_y = buffer_df.loc[:,['sleep_quality']]

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(trainSet_X, trainSet_y, test_size=0.2, shuffle=True)

# Dataframe to numpy array
buffer_y_train = y_train.values
buffer_y_test = y_test.values
# float to int
buffer_y_train = buffer_y_train.astype(int)
buffer_y_test = buffer_y_test.astype(int)

buffer_y_train = buffer_y_train 
buffer_y_test = buffer_y_test 

buffer_y_train = buffer_y_train.T
buffer_y_test = buffer_y_test.T

#y_train = convert_to_one_hot(buffer_y_train,3)
#y_test = convert_to_one_hot(buffer_y_test,3)

# y_train = buffer_y_train
# y_test = buffer_y_test

#X_train = X_train.T
#X_test = X_test.T

# from dataframe to numpy array
X_train = X_train.values
X_test = X_test.values

y_train = y_train.astype(int)
y_test = y_test.astype(int)

train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)
params={
    'boosting_type': 'gbdt',
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':7,
    'num_leaves': 100,
    'objective':'multiclass',
    'num_class':3,
}
clf=lgb.train(params,train_data,valid_sets=[validation_data])
y_pred=clf.predict(X_test)
#for i in range(len(y_pred)):
#  if(y_pred[i] > 0.5):
#    y_pred[i] = 1
#  else:
#    y_pred[i] = 0
y_pred=[list(x).index(max(x)) for x in y_pred]
print('accuracy_score')
print(accuracy_score(y_test,y_pred))
print('recall_score')
print(metrics.recall_score(y_test,y_pred,average='macro'))
print('precison_score')
print(metrics.precision_score(y_test,y_pred,average='macro'))
print('F1_score')
print(metrics.f1_score(y_test,y_pred,average='macro'))
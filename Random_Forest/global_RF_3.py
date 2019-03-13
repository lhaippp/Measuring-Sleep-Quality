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
#from tf_utils import random_mini_batches,convert_to_one_hot

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

clf = RandomForestClassifier(n_estimators = 300)
model = clf.fit(X_train , y_train)
r_train = clf.score(X_train , y_train)
print(r_train)
r_test = clf.score(X_test , y_test)
print(r_test)
y_pred = clf.predict(X_test)
cm=confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
labels = ['Bad Sleep', 'Ordinary Sleep', 'Good Sleep']
tick_marks = np.array(range(len(labels))) + 0.5
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=80)
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='magenta', fontsize=15, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.show()

y_pred=clf.predict(X_test)
print('accuracy_score')
print(accuracy_score(y_test,y_pred))
print('recall_score')
print(metrics.recall_score(y_test,y_pred,average='macro'))
print('precison_score')
print(metrics.precision_score(y_test,y_pred,average='macro'))
print('F1_score')
print(metrics.f1_score(y_test,y_pred,average='macro'))
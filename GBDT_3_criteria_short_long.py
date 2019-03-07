import pandas as pd 
import numpy as np
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn import metrics


buffer_df = pd.read_csv('SleepQuality_DNN_Classification_3Critere.csv')
# buffer_df = pd.read_csv('SleepQuality_DNN_Classification_binary.csv')

# select features and calculate their coefficients
corr = buffer_df[['avg23bpd_s2','avg23bps_s2','ai_all','rdi0p', 'nsupinep', 'pctlt75', 'pctlt80', 'pctlt85', 'pctlt90',
                     'slp_eff', 'slp_lat', 'slpprdp', 'supinep', 'times34p', 'timest1p', 'timest2p', 'waso']+['ms204c']].corr()['ms204c'].abs().sort_values(ascending = False)

cols = corr.index.values.tolist()[1:18]
trainSet_X = buffer_df.loc[:,cols]

# Sleep Quality from light to dark: 1 -> 5
# trainSet_y = buffer_df.loc[:,['ms204a']]
trainSet_y = buffer_df.loc[:,['ms204b']]
# trainSet_y = buffer_df.loc[:,['ms204c']]

# split training set and test set
X_train, X_test, y_train, y_test = train_test_split(trainSet_X, trainSet_y, test_size=0.2, shuffle=True)

# Dataframe to numpy array
buffer_y_train = y_train.values
buffer_y_test = y_test.values

# float to int
buffer_y_train = buffer_y_train.astype(int)
buffer_y_test = buffer_y_test.astype(int)

buffer_y_train = buffer_y_train.T
buffer_y_test = buffer_y_test.T


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
    'objective':'binary',
}

clf=lgb.train(params,train_data,valid_sets=[validation_data])

y_pred=clf.predict(X_test)
for i in range(len(y_pred)):
  if(y_pred[i] > 0.5):
    y_pred[i] = 1
  else:
    y_pred[i] = 0
print('accuracy_score')
print(accuracy_score(y_test,y_pred))
print('recall_score')
print(metrics.recall_score(y_test,y_pred,average='macro'))
print('precison_score')
print(metrics.precision_score(y_test,y_pred,average='macro'))
print('F1_score')
print(metrics.f1_score(y_test,y_pred,average='macro'))
#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np

import pandas as pd
from pandas import DataFrame, Series
from sklearn import datasets
from matplotlib import pyplot as plt

from xgboost import cv as xg_cv, DMatrix, train as xg_train
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE, recall_score, roc_auc_score, accuracy_score, confusion_matrix
import xgboost
import sys
import joblib

# # 构造数据集

# In[9]:


X, y = datasets.make_blobs(n_samples=[9000, 1000], n_features=2, centers=[[0, 0], [5, 5]], cluster_std=[2, 3],
                           random_state=1)
X = DataFrame(X, columns=['f1', 'f2'])
X['flag'] = y
X

# # 查看数据分布

# In[10]:


ax = plt.gca()

X[X['flag'] == 0].plot.scatter(x='f1', y='f2', c='red', ax=ax)
X[X['flag'] == 1].plot.scatter(x='f1', y='f2', c='green', ax=ax)

# # 使用xgboost对数据进行分类

# In[32]:


param_1 = {'random_state': 1,
           'silent': False,
           'objective': 'binary:logistic',
           'eval_metric': 'auc',  # logloss
           'eta': 0.01,
           'max_depth': 2,
           'booster': 'gbtree',
           'alpha': 0,
           'lambda': 1,
           'gamma': 1,
           'colsample_bytree': 1,
           'subsample': 0.5,
           'scale_pos_weight': 1

           }

param_2 = {'random_state': 1,
           'silent': False,
           'objective': 'binary:logistic',
           'eval_metric': 'auc',  # logloss
           'eta': 0.01,
           'max_depth': 2,
           'booster': 'gbtree',
           'alpha': 0,
           'lambda': 1,
           'gamma': 1,
           'colsample_bytree': 1,
           'subsample': 0.5,
           'scale_pos_weight': 1
           }
num_boost_round = 400;
ret_score_1 = xg_cv(params=param_1, dtrain=DMatrix(data=X.iloc[:, :-1], label=X.iloc[:, -1]),
                    num_boost_round=num_boost_round, nfold=5)

ret_score_2 = xg_cv(params=param_2, dtrain=DMatrix(data=X.iloc[:, :-1], label=X.iloc[:, -1]),
                    num_boost_round=num_boost_round, nfold=5)

# In[33]:


fig, ax = plt.subplots(1, 1)
ret_score_1.iloc[:, [0, 2]].plot(ax=ax, legend=True, color=['black', 'green'])
ret_score_2.iloc[:, [0, 2]].plot(ax=ax, legend=True, color=['red', 'grey'])

print((ret_score_1.iloc[:, 2] - ret_score_1.iloc[:, 0]).max())
print((ret_score_2.iloc[:, 2] - ret_score_2.iloc[:, 0]).max())

# # 使用条好的参数建模

# In[34]:


param = {'random_state': 1,
         'silent': False,
         'objective': 'binary:logistic',
         'eval_metric': 'auc',  # logloss
         'eta': 0.01,
         'max_depth': 2,
         'booster': 'gbtree',
         'alpha': 0,
         'lambda': 0,
         'gamma': 1,
         'colsample_bytree': 1,
         'subsample': 0.5,
         'scale_pos_weight': 1,
         'num_boost_round': 400
         }

train, test = TTS(X, test_size=0.5, random_state=2)
xgb = xgboost.train(param, dtrain=DMatrix(train[['f1', 'f2']], label=train['flag']))

# In[35]:


pre_ret = xgb.predict(DMatrix(test[['f1', 'f2']], label=test['flag']))
threshold = np.linspace(0, 1, 50)

recall = []
c_m = []

for thres in threshold:
    pre_ret_copy = pre_ret.copy()
    pre_ret_copy[pre_ret_copy > thres] = 1;
    pre_ret_copy[pre_ret_copy != 1] = 0;
    c_m.append(confusion_matrix(test['flag'], pre_ret_copy, labels=[1, 0]))
    recall.append(recall_score(test['flag'], pre_ret_copy, labels=[1, 0]))

print('auc: ', roc_auc_score(test['flag'], pre_ret))

plt.scatter(threshold, recall)

plt.show()
# # 查看特征重要性

# In[43]:


xgb.get_fscore()

# # 绘制特征重要性

# In[44]:


xgboost.plot_importance(xgb)

# # 保存模型

# In[45]:


joblib.dump(value=xgb, filename='./model/xgb_makeblobs.model')

# In[ ]:





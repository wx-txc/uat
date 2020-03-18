#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[14]:


data = pd.read_csv(r'D:\study\python3\data\titanic.csv', sep=',')
data.head()


# # 删除没用的特征

# In[15]:


data = data.drop(columns = ['PassengerId', 'Name', 'Ticket'])
data.head()


# In[16]:


data.info()


# # 删除缺失值太多的列

# In[17]:


data = data.drop(columns = ['Cabin'])
data.info()


# # 给age填补缺失值

# In[18]:


age_mean = data['Age'].mean();
data['Age'] = data['Age'].apply(lambda x:age_mean if np.isnan(x) else x)
data.info()


# # 删除Embarked为空的行

# In[19]:


data = data.dropna(axis = 'index', how = 'any')
data.info()


# # 删除行后，要回复索引

# In[29]:


data.index = range(0, data.shape[0])
data.head()


# # 将Sex属性转化成分类属性

# In[30]:


# data['Sex'].value_counts()
sex_list = data['Sex'].unique().tolist()

data['Sex'] = data['Sex'].apply(lambda x: sex_list.index(x))
data.head()


# # 将Embarked进行OneHotEncoding

# In[36]:


# 实例化OneHot编码器并训练
enc = OneHotEncoder(categories = 'auto')
enc.fit(np.array(data['Embarked'])[:,np.newaxis])

# 进行特征OneHot编码
Embarked_feature_name = ['Embarked_'+i for i in data['Embarked'].unique()]
Embarked = pd.DataFrame(enc.transform(np.array(data['Embarked'])[:,np.newaxis]).toarray(), columns = Embarked_feature_name)
data_merge = pd.concat(objs = [data,Embarked],
                 axis = 'columns',
                 ignore_index=False
                )


# # 给标签进行编码

# In[39]:


y = data_merge['Survived']
le = LabelEncoder()
le = le.fit(y)
y = le.transform(y)
le.classes_


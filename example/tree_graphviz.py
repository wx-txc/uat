#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split as TTS
import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
import graphviz
import numpy as np
from collections import Counter

# In[2]:


data_wine = load_wine()
data_wine.keys()

# In[3]:


X = pd.DataFrame(data_wine.data, columns=data_wine.feature_names)
y = pd.DataFrame(data_wine.target, columns=['flag'])
pd.concat([X.head(), y.head()]).shape

# In[4]:


x_train, x_test, y_train, y_test = TTS(X, y, test_size=0.3, random_state=1)

# In[5]:


dtc = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=1, )
dtc.fit(x_train, y_train)
dtc.score(x_test, y_test)

# In[6]:


plt.barh(X.columns, dtc.feature_importances_, )

# In[8]:


dot_data = tree.export_graphviz(decision_tree=dtc,
                                feature_names=X.columns,
                                class_names=['清酒', '二锅头', '茅台'],
                                filled=True,
                                rounded=True,
                                #    out_file = 'dot_data',
                                )
# 因为有中文，所以需要修改字体类型
dot_data = dot_data.replace('helvetica', '"Microsoft YaHei"')
graph = graphviz.Source(source=dot_data, filename='tree_01', directory='./model', format='pdf')

# graph.view()  # 保存并在图片或pdf中打开，不在jupyter中打开

graph.render(view=False)

# In[12]:


'''
画出决策树后可以观察到，每个节点有一个value值，这个值的计算方式:
value = [class1的样本点个数 * 对应的权重，class2的样本点个数 * 对应的权重， class3的样本点个数 * 对应的权重 ]

min_weight_fraction_leaf 这个参数表示叶子节点的 sum(叶子:value)/ sum(根:value) > min_weight_fraction_leaf;
其中 sum(*: value) 表示节点*的总value值。

'''
dtc = tree.DecisionTreeClassifier(criterion='gini',
                                  splitter='random',
                                  random_state=0,
                                  max_depth=5,
                                  max_features=0.8,
                                  min_samples_leaf=1,
                                  min_samples_split=5,
                                  min_impurity_decrease=0.02,

                                  # 在决定某个叶子节点是哪一类型的时候，0类的个数乘以2，1类的个数乘以3， 2类的个数乘以4
                                  class_weight={0: 2, 1: 3, 2: 4},
                                  min_weight_fraction_leaf=0.1,  # 叶子节点的总value值与根节点的value值的比要大于这个值
                                  )
dtc.fit(x_train, y_train)
print(dtc.score(x_test, y_test))
plt.barh(X.columns, dtc.feature_importances_, )

dot_data = tree.export_graphviz(decision_tree=dtc,
                                feature_names=X.columns,
                                class_names=['清酒', '二锅头', '茅台'],
                                filled=True,
                                rounded=True,
                                #    out_file = 'dot_data',
                                )

# 因为有中文，所以需要修改字体类型
dot_data = dot_data.replace('helvetica', '"Microsoft YaHei"')

# 参考例子 https://www.cnblogs.com/Zzbj/p/11431015.html
graph = graphviz.Source(source=dot_data, filename='tree_02', directory='./model', format='png', )
graph.render()
print(dtc.apply(x_test))  # 返回每个样本点在决策树叶子节点的索引
plt.show()

# In[11]:


# In[ ]:





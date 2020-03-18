#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
from matplotlib import pyplot as plt, markers, colors, cm
from sklearn import datasets
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as TTS

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# # 准备数据

# In[2]:


data = [make_moons(n_samples=300, random_state=1, noise=0.2),
        make_circles(n_samples=300, random_state=1, noise=0.1),
        make_classification(n_samples=300, n_features=2, n_redundant=0, random_state=1, )
        ]

# # 画图

# In[3]:


fig, axes = plt.subplots(3, 2)
font = {  # 'family':'serif',
    'style': 'italic',
    'weight': 'normal',
    'color': 'purple',
    'size': 20
}

font2 = {  # 'family':'serif',
    'style': 'italic',
    'weight': 'normal',
    'color': 'black',
    'size': 10
}
fig.set_size_inches(10, 15)
for i in range(3):
    # 设置坐标轴的范围
    x, y = data[i]
    min_x = min(x[:, 0]) - 0.5;
    max_x = max(x[:, 0]) + 0.5;
    min_y = min(x[:, 1]) - 0.5;
    max_y = max(x[:, 1]) + 0.5;

    axes[i][0].set_xlim((min_x, max_x))
    axes[i][0].set_ylim((min_y, max_y))
    axes[i][1].set_xlim((min_x, max_x))
    axes[i][1].set_ylim((min_y, max_y))

    axes[i][0].set_xticks(())
    axes[i][0].set_yticks(())
    axes[i][1].set_xticks(())
    axes[i][1].set_yticks(())

    # 设置标题
    if (i == 0):
        axes[i][0].set_title(label='分类前', pad=10, fontdict=font)
        axes[i][1].set_title(label='分类后', pad=10, fontdict=font)

    # 生成网格点
    array1, array2 = np.meshgrid(np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y, 0.1))

    # 数据 TTS
    x_train, x_test, y_train, y_test = TTS(x, y, test_size=0.3)

    # 画点
    axes[i][0].scatter(x=x_train[:, 0],
                       y=x_train[:, 1],
                       c=y_train,
                       marker='o',
                       s=30,
                       cmap=colors.ListedColormap(['red', 'blue']),
                       edgecolor='black'
                       )

    # 训练数据
    clf = DecisionTreeClassifier(criterion='entropy',
                                 random_state=1,
                                 splitter='best',
                                 max_depth=4
                                 ).fit(x_train, y_train)

    Z = clf.predict_proba(np.c_[array1.ravel(), array2.ravel()])[:, 1];
    Z = Z.reshape(array1.shape)

    # 画等高线
    axes[i][1].contourf(array1, array2, Z, cmap='RdBu', alpha=.8)  # cmap: RdBu, BrBG

    # 画点
    axes[i][1].scatter(x=x_test[:, 0],
                       y=x_test[:, 1],
                       c=y_test,
                       marker='o',
                       s=30,
                       cmap=colors.ListedColormap(['red', 'blue']),
                       edgecolor='black'
                       )

    # 添加测试集准确率
    axes[i][1].text(x=min_x + 2 / 3 * (max_x - min_x),
                    y=min_y + 5 / 6 * (max_y - min_y),
                    s='准确率：{:.2f}'.format(clf.score(x_test, y_test)),
                    fontdict=font2
                    )

fig.show()
# In[ ]:





# coding:utf-8
'''
this is an example for multiple linear regression
'''

import numpy as np;
from sklearn import tree
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from matplotlib import pyplot as plt


# 准备数据
x = np.random.uniform(0, 10, 500).reshape(-1,1);
y = np.sin(x) + np.random.normal(0,0.5,len(x)).reshape(-1,1) / 5;

# 划分测试集、训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# 通过散点图查看数据分布
fig, ax = plt.subplots(1,1)
ax.scatter(x, y)


# 使用线性回归器进行拟合
dtr = tree.DecisionTreeRegressor().fit(x_train,y_train);
linearR = LinearRegression().fit(x_train, y_train);

# 查看预测结果分布
plt.scatter(x_test+15, dtr.predict(x_test), c = 'red')
plt.scatter(x_test+15, linearR.predict(x_test), c = 'blue')


# 使用分箱技术拟合
n_bins = 10
kbder = KBinsDiscretizer(n_bins=n_bins, strategy = 'uniform').fit(x_train)


# 使用线性分类器在分箱后数据上拟合
lr_kbder = LinearRegression().fit(X = kbder.transform(x_train).toarray(), y = y_train)

plt.grid()
plt.scatter(x_test, lr_kbder.predict(kbder.transform(x_test).toarray()))
plt.scatter(x_test, y_test)
plt.vlines(x=kbder.bin_edges_[0], ymin = -1, ymax = 1,)

# 画概率分布直方图与密度函数
normal_x = np.random.normal(10, 1, 1000);
sns.distplot(normal_x, bins = 20)

# 画二维码
plt.figure(figsize = (5, 0.5))
sns.rugplot(normal_x, height = 1)

plt.show()
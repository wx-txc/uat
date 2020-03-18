# coding: utf-8
'''
This example is creating a xgboost model on boston dataset.
'''
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split as TTS, cross_val_score as CVS, KFold, learning_curve, \
    cross_validate
from sklearn.metrics import mean_squared_error as mse
import xgboost
from xgboost import XGBRegressor, DMatrix
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 到如波斯顿房价数据集
data = load_boston()

X = data.data
y = data.target

# 划分测试集训练集
x_train, x_test, y_train, y_test = TTS(X, y, test_size=0.3)

# 初始化xgbooster模型并训练
xgbr = XGBRegressor(n_estimators=100).fit(X=x_train, y=y_train)

# 使用训练好的模型在测试集上预测
y_test_pre = xgbr.predict(x_test)

print('r2: ', xgbr.score(x_test, y_test))
print('mse: ', mse(y_test_pre, y_test))

# # 画学习曲线
xgbr = XGBRegressor(n_estimators=100)
cv = KFold(n_splits=5, shuffle=True, random_state=23)

# 使用seaborn画图
sns.set()
sns.color_palette('bright')
pal_style = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

# 绘制训练样本容量的学习曲线
train_size, train_scores, test_scores = learning_curve(xgbr, X=X, y=y, cv=cv, random_state=12)
sns.lineplot(train_size, train_scores.mean(axis=1), marker='o', color='red')
sns.lineplot(train_size, test_scores.mean(axis=1), marker='o', color='green')


# 绘制基分类器个数的学习曲线
x_axis = range(10, 1010, 50)
cv = KFold(n_splits=5, shuffle=True, random_state=45)
v_bias = [];
v_vars = [];
v_error = [];
for i in x_axis:
    xgbr = XGBRegressor(n_estimators=i);
    score = CVS(xgbr, X, y, cv=cv);
    v_bias.append(score.mean())
    v_vars.append(score.var())
    v_error.append((1 - score.mean()) ** 2 + score.var())

print(max(v_bias), x_axis[v_bias.index(max(v_bias))], v_error[v_bias.index(max(v_bias))])

print(min(v_vars), x_axis[v_vars.index(min(v_vars))], v_error[v_vars.index(min(v_vars))])

print(min(v_error), x_axis[v_error.index(min(v_error))], v_bias[v_error.index(min(v_error))])

# 绘制偏差、方差学习曲线
sns.lineplot(x_axis, v_bias, marker='o', color='red')
sns.lineplot(x_axis, np.array(v_bias) - np.array(v_vars), marker='+', color='green', )
sns.lineplot(x_axis, np.array(v_bias) + np.array(v_vars), marker='+', color='green', )


def eta_learning_curve(etas, ax, score_list, cv):
    score_means = []
    for i in etas:
        xgbr = XGBRegressor(n_estimator=40, learning_rate=i, silent=True)
        scores = cross_validate(xgbr, X, y, cv=cv, scoring=score_list)
        score_mean = []
        score_means.append(score_mean)
        for score_name in score_list:
            score_mean.append(scores['test_' + score_name].mean())

    # print(np.array(score_means).shape)
    ret_df = pd.DataFrame(data=score_means, columns=score_list)
    ret_df['eta'] = etas;

    return ret_df;


# In[13]:


cv = KFold(n_splits=5, shuffle=True, random_state=2)
score_list = ['r2', 'neg_mean_squared_error']
etas = np.linspace(0.01, 0.9, 10)
ret = eta_learning_curve(etas, plt.gca, score_list, cv)

# In[14]:


ret.plot(x='eta', y='r2')

# In[15]:


booster = ['dart', 'gbtree', 'gblinear']
cv = KFold(n_splits=5, random_state=3, shuffle=True)

# In[16]:


CVS(XGBRegressor(n_estimator=40, booster='dart', slient=True, learning_rate=0.15), cv=cv, X=X, y=y, scoring='r2').mean()

# In[17]:


CVS(XGBRegressor(n_estimator=40, booster='gbtree', silent=True, learning_rate=0.15), cv=cv, X=X, y=y,
    scoring='r2').mean()

# In[18]:


CVS(XGBRegressor(n_estimator=40, booster='gblinear', silent=True, learning_rate=0.15), cv=cv, X=X, y=y,
    scoring='r2').mean()

# In[142]:


diabetes = datasets.load_diabetes()
dia_X = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
dia_y = diabetes.target
dia_X.head()
dia_y[0:10]

params = {'subsample': 0.9,
          'silent': False,
          'eta': 0.2,
          'alpha': 0.1,
          'gamma': 0.1,
          'lambda': 0.6,
          'booster': 'dart',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'importance_type': 'gain'
          }
# xgboost.train(params=params, dtrain = DMatrix(dia_X, dia_y), num_boost_round= 100,)

scores = xgboost.cv(params=params, dtrain=DMatrix(dia_X, dia_y), num_boost_round=100, nfold=5)
scores[['train-rmse-mean', 'test-rmse-mean']].plot()


plt.show()


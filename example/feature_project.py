from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectFromModel, RFE
from sklearn.model_selection import cross_val_score as cvs
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt


data = pd.read_csv(r'../data/digit_recognizor.csv')

'''
方差过滤法：过滤掉方差小于一定阈值的特征。和标签无关
此方法对knn，svm，逻辑回归，回归 等需要遍历特征的模型比较使用，主要是用来删除无用特征以减少算法的运行时间。
对随机森林无效，因为随机森林本来就是随机选取部分特征。在sklearn中，单科决策树也是采用随机选择部分特征进行节点的划分。
'''
threshold = 0.1
vt_filter = VarianceThreshold(threshold=threshold)
vt_filter.fit(data)
remove_features = [i for i,j in zip(data.columns, vt_filter.variances_) if j <= threshold]
print(remove_features)
data_new = vt_filter.transform(data.iloc[10000:,:])


'''
统计量过滤法：
chi2: 卡方检验选择特征，只能做离散型。
f_classif: f检验选择分类特征
mutual_info_classif: 互信息选择特征

以上选择特征的量都是用来检验特征与标签之间的相关性强度。
'''
SelectKBest(score_func=chi2, k = 400).fit_transform(X = data_new[:,1:], y = data_new[:,0])

time_0 = time.clock()
chi2(X = data_new[:,1:], y = data_new[:,0])
time_1 = time.clock()
print('chi2: {:2f}'.format(time_1-time_0))
f_classif(X = data_new[:,1:], y = data_new[:,0])
time_2 = time.clock()
print('f: {:2f}'.format(time_2-time_1))

# mutual_info_classif(X = data_new[:,1:], y = data_new[:,0])
# time_3 = time.clock()
# print('mutual_info: {:2f}'.format(time_3-time_2))

'''
嵌入法(embeded)特征选择：
SelectFromModel: 使用模型参数的重要性选择特征，凡是模型有feature_importances_ 或者 coef_ 属性的都可以与SelectFromModel
结合使用。
'''
# 训练一个随机森林模型
rfc_ = RandomForestClassifier(n_estimators=10, random_state=0)
feature_importances = rfc_.fit(data.iloc[:, 1:], data.iloc[:, 0]).feature_importances_

scores = []

# 画阈值的学习曲线
x_axis = np.linspace(0, feature_importances.max(), 20);
for i in np.linspace(0, feature_importances.max(), 20):
    data_embedded = SelectFromModel(estimator=rfc_, threshold=i, prefit=True).transform(data.iloc[:, 1:])
    scores.append(cvs(rfc_, data_embedded, data.iloc[:, 0], cv = 5).mean())
    print(scores)

fig = plt.gcf()
ax = plt.gca()
ax.plot(x_axis, scores)
fig.show()


'''
使用包装法（wrapper）进行特征选择：
迭代的训练模型，训练模型后，根据模型的feature_importances_或coef_来删除部分特征，使用剩余特征继续建模，继续删...
接口：feature_selection.RFE 或 RFECV
'''
rfc = RandomForestClassifier(n_estimators=10, random_state=0)
score_rfe = []
fig = plt.gcf()
ax = plt.gca()
for i in range(1, 400, 20):
    x_wrapper = RFE(estimator=rfc, n_features_to_select=100, step=10).fit_transform(data.iloc[:, 1:], data.iloc[:, 0])
    score_rfe.append(cvs(rfc, x_wrapper, data.iloc[:, 0]).mean())

    print(score_rfe)

ax.plot(range(1, 400, 20), score_rfe)
fig.show()


X = pd.DataFrame(data.data, columns = data.feature_names)
y = data.target
X.describe
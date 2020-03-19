from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest
import numpy as np
import pandas as pd
import time
import datetime


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
mutual_info_classif(X = data_new[:,1:], y = data_new[:,0])
time_3 = time.clock()

print('mutual_info: {:2f}'.format(time_3-time_2))
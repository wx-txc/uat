from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SS, LabelEncoder, OrdinalEncoder, \
    OneHotEncoder, Binarizer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

'''
无量纲化：归一化、标准化
主要思想：中心化（zero-centered or subtraction）和缩放（scale）
'''
# Normalization: 归一化
x = pd.DataFrame(data=[[1,3,5],[2,4,6]]).T
mms = MMS(feature_range=(0,1))
mms.fit(x)
x_nor = mms.transform(x)
x_inverse = mms.inverse_transform(x_nor)

# Standardization(Z-Score Normalization): 标准化
ss = SS().fit(x)
x_std = ss.transform(x)
print(ss.mean_, ss.var_)


'''
缺失值填补：sklearn.impute.SampleImputer
'''
titanic = pd.read_csv(r'../data/titanic.csv', sep=',')
titanic.info()
imp_median = SimpleImputer(missing_values=np.nan,
                           strategy='median',
                           copy = True
                          )

titanic['Age'] = imp_median.fit_transform(titanic['Age'].values.reshape((-1,1)))
titanic.info()
imp_mode = SimpleImputer(strategy='most_frequent',copy = True)
titanic['Embarked'] = imp_mode.fit_transform(titanic['Embarked'].values.reshape((-1,1)))
titanic.info()
titanic = titanic.drop(columns = ['Cabin'],inplace = False)
titanic.info()


'''
变量编码：
类别标签：LabelEncoder
无序特征编码：OneHotEncoder
有序离散特征编码：OrdinalEncoder
'''
titanic = titanic[['Sex', 'Age', 'Embarked','Pclass', 'Survived']]
titanic.head()

enc = OneHotEncoder(categories = 'auto')
oneHot_ret = enc.fit_transform(titanic[['Sex','Embarked']]).toarray()

print(enc.get_feature_names())

titanic_new = pd.concat([titanic, pd.DataFrame(oneHot_ret, columns = enc.get_feature_names())], axis = 'columns').drop(['Sex', 'Embarked'], axis='columns')
titanic_new


x = pd.DataFrame(np.random.uniform(1,20,20).reshape((-1,2)), columns = ['x','y'])
bin = Binarizer(threshold = 10)
x_bin = bin.fit_transform(x)

kbd = KBinsDiscretizer(n_bins = 5, encode='onehot-dense', strategy='kmeans')
x_kbd = kbd.fit_transform(x)
print(x_kbd)
kbd.n_bins_
kbd.bin_edges_


from sklearn.linear_model import Ridge, Lasso, ElasticNet
Ridge()
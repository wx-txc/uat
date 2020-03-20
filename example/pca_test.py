from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# 准备数据
data = load_iris()
X = pd.DataFrame(data.data, columns = data.feature_names)
y = data.target
print('各维度的标准差: ',X.std())

# 协方差
cov = np.cov(X, rowvar = False)
hmap = sns.heatmap(cov, xticklabels=data.feature_names, yticklabels = data.feature_names, cmap = 'BrBG',
            robust = False, annot = True, vmin = 0, vmax = 6,linewidths = 2
           )
top, bottom = hmap.get_ylim()
hmap.set_ylim(bottom-0.5, top+0.5)
plt.show()

# 降维
pca = PCA(n_components=0.99, svd_solver='full')
x_re = pca.fit(X)
print('主成分：',pca.components_)
print('每个主成分的方差：',pca.explained_variance_)
print('每个主成分在原特征方差上的贡献：',pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
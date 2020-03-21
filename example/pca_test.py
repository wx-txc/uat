from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, colors

# 字体使用SimHei，支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False
print(plt.rcParams)
# font.sans-serif: ['SimHei', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif']

# 准备数据
data = datasets.load_iris()
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

'''
PCA说明：
1. 在sklearn中都是采用SVD分解来计算主成分的。不采用求解协方差矩阵的特征值的方式。
2. SVD分解有两种常用的形式，紧奇异值分解 和 截断奇异值分解
3. sklearn.decomposition.PCA类参数svd_solver的四个取值解析：
    auto :
        the solver is selected by a default policy based on `X.shape` and
        `n_components`: if the input data is larger than 500x500 and the
        number of components to extract is lower than 80% of the smallest
        dimension of the data, then the more efficient 'randomized'
        method is enabled. Otherwise the exact full SVD is computed and
        optionally truncated afterwards.
    full :
        run exact full SVD calling the standard LAPACK solver via
        `scipy.linalg.svd` and select the components by postprocessing。
        完全奇异值分解，适合小样本。
    arpack :
        run SVD truncated to n_components calling ARPACK solver via
        `scipy.sparse.linalg.svds`. It requires strictly
        0 < n_components < min(X.shape)
        这种模式适合大样本，多特征的情况。需要一个随机数种子
    randomized :
        采用Halko等人提出的随机奇异值分解算法进行SVD分解。优点是运算速度快。需要一个随机数种子。

'''
pca = PCA(n_components=2, svd_solver='full')
x_re = pca.fit_transform(X)
print('主成分：',pca.components_)
print('每个主成分的方差：',pca.explained_variance_)
print('每个主成分在原特征方差上的贡献：',pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

ax = plt.gca()
plt.scatter(x = x_re[:,0],
            y = x_re[:,1],
            c = y,
            cmap=colors.ListedColormap(['red','green','grey']),
            label = '点'
            )
plt.legend()
plt.show()
data = datasets.fetch_lfw_people()

pca.inverse_transform()
pca.fit()

'''
pca图像识别
'''

# 定义一个画图函数
def plot_image(images, n_rows, n_cols, cmap='Greys'):
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(10, 5),
                             subplot_kw={
                                 'xticks': [],
                                 'yticks': []
                             });
    axes = axes.flat;
    for i in range(n_rows * n_cols):
        axes[i].imshow(images[i], cmap=cmap)

    fig.show()

# 获取人脸图像
data = datasets.fetch_lfw_people()
data.keys()

# 取出数据
X = data.data
y = data.target
images = data.images
print('X.shape: {}, images.shape: {}'.format(X.shape, images.shape))

# 画出人脸图
plot_image(images, 2,4, cmap = 'Greys')

# pca降维
pca = PCA(n_components=200, svd_solver = 'full',whiten = True).fit(X)
X_de = pca.transform(X)
X_de.shape

# 查看主成分
principal_components = pca.components_
principal_components.shape

# 查看主成分的解释方差
pca.explained_variance_ratio_.sum()

# 查看主成分的图像
plot_image(principal_components.reshape(-1,62,47), 2,5)

# 重构原特征
X_reverse = pca.inverse_transform(X_de)
X_reverse.shape


# 获取原数据特征的协方差
pca.get_covariance()
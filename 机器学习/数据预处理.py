from sklearn.model_selection import train_test_split
import pandas as pd
import os
os.chdir(r'/mnt/d/售后/机器学习/dataset')

df_wine = pd.read_csv('wine.csv')
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# step1. 特征标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_standarscale = sc.fit_transform(X_train)
X_test_standarscale = sc.fit_transform(X_test)

# step2. 构造协方差矩阵
import numpy as np
cov_matrix = np.cov(X_train_standarscale.T)

# step3. 计算协方差矩阵的特征值和特征向量，使用linalg.eig函数对协方差矩阵进行特征分解
# 产生了一个由13个特征值组成的向量（eigen_vals），对应的特征向量按列存储在13x13的矩阵中（eigens_vecs）
eigen_vals, eigens_vecs = np.linalg.eig(cov_matrix)

# 绘制方差解释比
total = sum(eigen_vals)
var_exp = [(i/total) for i in sorted(eigen_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp) # cumsum计算方差解释的累积

import matplotlib.pyplot as plt
feature_num = df_wine.shape[1]
plt.bar(range(1, feature_num), var_exp, align='center', label = 'Individual explained variance')
plt.step(range(1, feature_num), cum_var_exp, where='mid', label = 'Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# step4.1 按照特征值降序对特征值-特征向量对 进行排序
# 构造 “特征值-特征向量对”
eigen_pairs = [(np.abs(eigen_vals[i]), eigens_vecs[:, i]) for i in range(len(eigen_vals))] 

# 从高到低排列
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# step4.2 由于要绘制二维散点图，因此选择两个特征向量，创建了一个13x2的投影矩阵
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
              eigen_pairs[1][1][:, np.newaxis]))

print(f'Matrix W:\n{w}')

# step4.3 使用投影矩阵将样本x变换到PCA子空间
X_train_pca = X_train_standarscale.dot(w)

# 可视化
colors = ['r', 'b', 'g']
makers = ['o', 's', '^']

for l, c, m in zip(np.unique(y_train), colors, makers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# 使用sckit-learn
# 评估特征的贡献
loadings = eigens_vecs * np.sqrt(eigen_vals)
# 绘制第一个主成分的载荷loadings[:,0]，此向量为载荷矩阵中的第一列
fig, ax = plt.subplots()
ax.bar(range(feature_num-1), loadings[:, 0], align = 'center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()




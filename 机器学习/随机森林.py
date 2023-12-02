from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt
import numpy as np
import os
os.chdir(r'/Users/yangpei/YangPei/machine_learning')

"""
分类器的计算性能、预测性能在很大程度上取决于训练数据。监督机器学习算法训练过程涉及如下五个主要步骤：
1. 选择特征并收集标记的训练样本
2. 选择机器学习算法性能度量指标
3. 选择机器学习算法并训练模型
4. 评估模型的性能
5. 更改算法参数并调优模型
"""

# ===============================================
#             ***   1.  随机森林   ***
# ===============================================


# ============= step1. 数据准备 =============
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
Y = iris.target

# 将数据集拆分为训练数据集和测试数据集。train_test_split函数将X和Y随机拆分为两部分，30%的作为测试集，70%作为训练集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y) # stratify表示支持按比例分层策略

# 许多机器学习算法和优化算法需要特征缩放以获得最佳性能(特征缩放并非决策树算法必需的要求)
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
"""

# numpy.vstack() 用于垂直堆叠数组，即沿着垂直方向将数组堆叠在一起。将两个或多个数组沿着它们的第一个轴（行方向）堆叠，创建一个更大的数组。
X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))


# sikit-learn 的metrics模块有大量衡量模型性能的指标
from sklearn.metrics import accuracy_score
# print('Accuracy: %.3f'% accuracy_score(Y_test, Y_pred))

# 设置超参数
from mlxtend.plotting import plot_decision_regions
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, Y_train)
plot_decision_regions(X_combined, Y_combined, clf=forest)
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 用随机森林评估特征的重要性
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, Y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print('{0}, {1}'.format(f+1, importances[indices[f]]))

# 重要性绘图
plt.title('feature importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tighe_layout()
plt.show()
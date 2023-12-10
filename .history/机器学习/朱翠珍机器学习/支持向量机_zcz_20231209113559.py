from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir(r'/Users/yangpei/YangPei/after_sale/朱翠珍--机器学习')
df = pd.read_excel('finnal_ml_input.xlsx')

# 随机森林不要求假设数据线性可分，因此无需进行归一化，直接划分数据集
x = df.iloc[:, 0:13023].values
y = df.iloc[:, -1].values

# ============== step.2 数据集划分 ==============
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y) # stratify表示支持按比例分层策略

# ============== step.3 构建pipline对象 ==============
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# 构建pipeline对象
pipe_svc = make_pipeline(SVC(random_state=1))


# 设置超参数并使用网络搜索法构建学习器
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]
















































"""
# 定义内循环的学习器，使用网络搜索法进行内部交叉验证，选择出最优参数模型
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, 
                  scoring= 'accuracy', cv=2, n_jobs=-1, refit=True)

# 定义外循环，将最佳型传递到外循环，以评估模型性能
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

scores = cross_val_score(gs, X_train, Y_train, scoring='accuracy', cv=10)
print(f'CV accuracy: {np.mean(scores):.3f}' f'+/-{np.std(scores):.3f}')


# 执行外循环 --- 遍历嵌套的交叉验证迭代
for train_ix, test_ix in cv_outer.split(x, y):
    # 获取训练集和测试集
    X_train, X_test = x[train_ix,:], x[test_ix,:]
    y_train, y_test = y[train_ix], y[test_ix]
    perm_importance = permutation_importance(gs.fit(X_train, y_train), X_test, y_test)
    print(perm_importance)
"""












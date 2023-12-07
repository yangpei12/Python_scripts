from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlxtend
import os

os.chdir(r'E:\售后\朱翠珍--机器学习')
df = pd.read_excel('finnal_ml_input.xlsx')

# 随机森林不要求假设数据线性可分，因此无需进行归一化，直接划分数据集
x = df.iloc[:, 0:13023].values
y = df.iloc[:, -1].values

# ============== step.2 数据集划分 ==============
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y) # stratify表示支持按比例分层策略

# ============== step.3 构建pipline对象 ==============
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 随机森林保存50个特征
pipe_svc = make_pipeline(SelectKBest(k=50), SVC(random_state=1))


# 设置超参数构建学习器
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring= 'accuracy', cv=10)

# 拟合数据构建模型
grid_search = gs.fit(X=X_train, y=Y_train)

# 使用模型进行预测


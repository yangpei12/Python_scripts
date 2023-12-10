from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 构建pipeline对象
pipe_svc = make_pipeline(SVC(random_state=1))

# 设置超参数并使用网络搜索法构建学习器
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]

# 创建分层K折交叉验证对象并执行超参数搜索
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, cv=stratified_kfold, n_jobs=-1, refit=True)

# 交叉验证评分器
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=gs, X=X_train, y=Y_train, 
                         scoring='accuracy', cv=stratified_kfold, n_jobs=-1)

# 交叉验证评分
print(f'CV accuracy: {np.mean(scores):.3f}' f'+/-{np.std(scores):.3f}')

# 使用测试集估计模型的性能
pipe_svc = pipe_svc.fit(X=X_train, y=Y_train)
print(f'Test accuracy: {gs.score(X_test, Y_test):.3f}')

# 模型评估--学习曲线
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_svc, X=X_train, y=Y_train, 
                                                        train_sizes = np.linspace(0.1, 1.0, 10), 
                                                        cv=stratified_kfold, n_jobs=-1)

train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o',markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std,train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,color='green', linestyle='--', marker='s', 
         markersize=5, label='Validation accuracy')

plt.fill_between(train_sizes,test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.show()


# 8.2 精确率和召回率
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import matthews_corrcoef

Y_pred = pipe_svc.predict(X_test)
pre_val = precision_score(y_true= Y_test, y_pred=Y_pred)
print(f'Precision: {pre_val:.3f}')

rec_val = recall_score(y_true= Y_test, y_pred=Y_pred)
print(f'Recall: {rec_val:.3f}')

f1_val = f1_score(y_true= Y_test, y_pred=Y_pred)
print(f'F1 Score: {f1_val:.3f}')

mcc_val = matthews_corrcoef(y_true= Y_test, y_pred=Y_pred)
print(f'MCC Score: {mcc_val:.3f}')










































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












import os
import pandas as pd

# 数据准备
os.chdir(r'/mnt/d/售后/朱翠珍--机器学习/mRNA')
df = pd.read_excel('lasso_select_feature_mRNA.xlsx')

# 随机森林不要求假设数据线性可分，因此无需进行归一化，直接划分数据集
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# ============== step.2 数据集划分 ==============
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y) # stratify表示支持按比例分层策略

# ============== step.3 构建pipline对象 ==============
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 构建pipeline对象
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, probability=True))

# 设置超参数并使用网络搜索法构建学习器
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]

# 创建分层K折交叉验证对象、创建超参数搜索对象
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, cv=stratified_kfold, n_jobs=4, refit=True)

# 在网络超参数中执行嵌套交叉验证
gs.fit(X_train, y_train)

# 获取最佳模型
best_model = gs.best_estimator_

# 使用最佳模型进行预测
accuracy = best_model.score(X_test, y_test)

# 使用shap导出重要程度
import shap
shap.initjs()
explainer = shap.KernelExplainer(model=best_model.predict, data=X)
shap_values = explainer.shap_values(X) 

# 生成单个样本的力图
shap.force_plot(explainer.expected_value, shap_values[0,:], 
                 X[0, :], show=False, matplotlib=True).savefig('scratch.png')
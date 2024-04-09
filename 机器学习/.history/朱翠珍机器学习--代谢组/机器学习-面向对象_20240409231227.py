# 面向对象编程--包含四种常用的算法：支持向量机、随机森林、xgboost和逻辑回归
# 导入基础包
import os
import sys
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# 数据准备
argvs = sys.argv
# workDir = os.chdir(argvs[1])
# inputData = pd.read_csv(argvs[2], sep='\t', header=0)
workDir = os.chdir(r'/mnt/d/售后/朱翠珍--机器学习/mRNA')
inputData = pd.read_csv('lasso_select_feature_mRNA.txt', sep='\t', header=0)

X = inputData.iloc[:, 0:-1].values
y = inputData.iloc[:, -1].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

class ml_algorithm:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def lasso(self):
        pass

    def lr(self):

        from sklearn.linear_model import LogisticRegression

        # 构建pipeline对象
        pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=1,max_iter=10000, penalty='l1',
                                           multi_class='ovr', solver='liblinear'))

        # 设置超参数并使用网络搜索法构建学习器
        param_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        param_grid = [{'logisticregression__C': param_range}]

        # 创建分层K折交叉验证对象、创建超参数搜索对象
        stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, cv=stratified_kfold, n_jobs=3, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_

        # 返回最佳模型
        return best_model


    def svm(self):

        from sklearn.svm import SVC

        # 构建pipeline对象
        pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, probability=True))

        # 设置超参数并使用网络搜索法构建学习器
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]

        # 创建分层K折交叉验证对象、创建超参数搜索对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, cv=stratified_kfold, n_jobs=4, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_

        # 返回最佳模型
        return best_model
    

    def rf(self):
        from sklearn.ensemble import RandomForestClassifier
        # 构建pipline对象
        pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1))

        # 构建超参数网络
        param_grid = [{'randomforestclassifier__max_depth': [3,4,5,6,7,8], 
               'randomforestclassifier__min_samples_split': [1,2], 
               'randomforestclassifier__min_samples_leaf': [1,2],
               'randomforestclassifier__n_estimators':[200,300,400,500]}]
        
        # 构建网络搜索对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=stratified_kfold, n_jobs=4, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_
        
        # 返回最佳模型
        return best_model
        
    def xgb(self):

        from xgboost import XGBClassifier

        # 构建pipeline对象
        pipe_xgb = make_pipeline(StandardScaler(), XGBClassifier(random_state=1))

        # 设置超参数并使用网络搜索法构建学习器
        param_grid = [{'xgbclassifier__learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05], 
                    'xgbclassifier__max_depth': [5,6,7,8,9,10], 
                    'xgbclassifier__min_child_weight': [1],
                    'xgbclassifier__booster': ['gbtree'],
                    'xgbclassifier__n_estimators':[200, 300, 400],
                    'xgbclassifier__reg_alpha':[5, 6, 7, 8]}]
        
        # 创建分层K折交叉验证对象、创建超参数搜索对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        gs = GridSearchCV(estimator=pipe_xgb, param_grid=param_grid, cv=stratified_kfold, n_jobs=4, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_
        
        # 返回最佳模型
        return best_model

    def confusion_matrix_est(self, best_model):
        from sklearn.metrics import confusion_matrix
        Y_pred = best_model.predict(self.X_test)
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=Y_pred)
        # 绘制混淆矩阵图
        fig, ax = plt.subplots(figsize = (2.5, 2.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=.3)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i,j], va = 'center', ha='center')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig('xgboost_confusion_matrix.pdf')
        return 'Done'
    
    def learning_curve_plot(self):
        pass

    def sha(self):
        pass
if __name__ == "__main__":
    init_model = ml_algorithm
    svm_model = init_model.svm()
    confusion_matrix = init_model.confusion_matrix_est(svm_model)
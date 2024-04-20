# 面向对象编程--包含四种常用的算法：支持向量机、随机森林、xgboost和逻辑回归
# 导入基础包
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# 数据准备
argvs = sys.argv
# workDir = os.chdir(argvs[1])
# inputData = pd.read_csv(argvs[2], sep='\t', header=0)
inputData = pd.read_csv('/Users/yangpei/YangPei/after_sale/ml/meta/meta_exp_matrix.txt', sep='\t', header=0)
X = inputData.iloc[:, 0:-1].values
y = inputData.iloc[:, -1].values

# 创建特征筛选类
class features_select_algorithm:
    def __init__(self, X_train, X_test, y_train, y_test):
        # 构造函数
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def lasso_algorithm(self, path):
        # lasso算法进行特征筛选
        from sklearn.linear_model import LassoCV
        from sklearn.linear_model import Lasso

        # 标准化
        scaler = StandardScaler().fit(self.X_train) 
        X_train = scaler.transform(self.X_train)

        # 使用alpha优化lasso
        # 创建交叉验证模型
        lasso_model = LassoCV(cv=10, random_state=0, max_iter=10000)

        # 在训练集上对模型进行交叉验证
        clf = lasso_model.fit(X_train, y_train)

        # 根据交叉验证得到的最佳alpha 获取最佳模型，
        best_model = Lasso(alpha=clf.alpha_)

        # 由于没有refit，因此需要使用最佳参数的模型再次拟合数据
        best_model.fit(X_train, y_train)

        # 筛选特征
        sfm = SelectFromModel(best_model, max_features=10000)
        sfm.fit_transform(X, y)
        col_index = [i for i, value in enumerate(list(sfm.get_support())) if value == True]
        selected_features = inputData.columns[col_index]  # 获取选定的特征

        # 绘制lasso path图
        plt.semilogx(clf.alphas_, clf.mse_path_, ":")
        plt.plot(
            clf.alphas_ ,
            clf.mse_path_.mean(axis=-1),
            "k",
            label="Average across the folds",
            linewidth=2)
        
        plt.axvline(
            clf.alpha_, linestyle="--", color="k", label="alpha: CV estimate")

        # 调整y轴范围以与第一个图层重叠
        ymin, ymax = plt.ylim()
        plt.ylim(min(ymin, 50000), max(ymax, 250000))

        plt.legend()
        plt.xlabel("alphas")
        plt.ylabel("Mean square error")
        plt.title("Mean square error on each fold")
        plt.axis("tight")
        plt.savefig('%s/lasso_path_plot.pdf'%path)

        return selected_features, clf.alpha_
    
    def rfe(self):
        from sklearn.feature_selection import RFECV
        from sklearn.feature_selection import RFE

        # 标准化
        pass

    def rfc(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel

        # 创建学习器
        clf = RandomForestClassifier(n_estimators=500, random_state=0)

        # 拟合数据
        clf = clf.fit(self.X_train, self.y_train)

        # 构建selectModel对象
        sfm = SelectFromModel(clf, prefit=True)

        # 拟合数据
        sfm.transform(self.X_train)

        col_index = [i for i, value in enumerate(list(sfm.get_support())) if value == True]
        selected_features = inputData.columns[col_index]  # 获取选定的特征
        
        return selected_features
        

# 创建机器学习类
class ml_algorithm:
    def __init__(self, X_train, X_test, y_train, y_test):
        # 构造函数
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        self.stratified_kfold = stratified_kfold

# ==========================================================================================
#                                         模型构建部分
# ==========================================================================================
    
    def lr(self):
        from sklearn.linear_model import LogisticRegression

        # 构建pipeline对象
        pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=1,max_iter=10000, penalty='l1',
                                           multi_class='ovr', solver='liblinear'))

        # 设置超参数并使用网络搜索法构建学习器
        param_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        param_grid = [{'logisticregression__C': param_range}]

        # 创建分层K折交叉验证对象、创建超参数搜索对象
        gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, cv=self.stratified_kfold, n_jobs=3, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_

        # 返回最佳模型
        return best_model, clf


    def svm(self):

        from sklearn.svm import SVC

        # 构建pipeline对象
        pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, probability=True))

        # 设置超参数并使用网络搜索法构建学习器
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]

        # 创建分层K折交叉验证对象、创建超参数搜索对象
        gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, cv=self.stratified_kfold, n_jobs=4, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_

        # 返回最佳模型及网络搜索对象
        return best_model, clf


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
        gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=self.stratified_kfold, n_jobs=4, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_

        # 返回最佳模型
        return best_model, clf

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
        gs = GridSearchCV(estimator=pipe_xgb, param_grid=param_grid, cv=self.stratified_kfold, n_jobs=4, refit=True)

        # 在训练集上执行网络搜索并拟合
        clf = gs.fit(X = self.X_train, y = self.y_train)

        # 导出最佳模型
        best_model = clf.best_estimator_

        # 返回最佳模型
        return best_model, clf

# ==========================================================================================
#                                         模型评估部分
# ==========================================================================================
    
    def best_params(self, grid_search):
        # 输出模型最佳参数
        model_best_params = "the model's Best params are {0}\n".format(grid_search.best_params_)
        report.write(model_best_params)
    
    def test_accurary(self, best_model):
        # 使用测试集估计模型的性能
        model_test_accuracy = "the model's test accuracy: {0:.3f}\n".format(best_model.score(self.X_test, self.y_test))
        report.write(model_test_accuracy)


    def confusion_matrix_est(self, best_model, path):
        # 绘制混淆矩阵图
        from sklearn.metrics import confusion_matrix
        Y_pred = best_model.predict(self.X_test)
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=Y_pred)
        fig, ax = plt.subplots(figsize = (2.5, 2.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=.3)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i,j], va = 'center', ha='center')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig('%s/confusion_matrix.pdf'%path)

    
    def cross_val_score(self, best_model):
        # 嵌套交叉验证评分器
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(estimator=best_model, X = self.X_train, y = self.y_train, 
                                scoring='accuracy', cv=10, n_jobs=4)
        cross_val_score_result = 'CV accuracy: {0:.3f}' f'+/-{1:.3f}\n'.format(np.mean(scores), np.std(scores))
        report.write(cross_val_score_result)

        
    def learning_curve_plot(self, best_model, path):
        # 绘制学习曲线
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(estimator = best_model, X = self.X_train, y = self.y_train, 
                                                                train_sizes = np.linspace(0.1, 1.0, 10), 
                                                                cv=self.stratified_kfold, n_jobs=4)

        train_mean = np.mean(train_scores,axis=1)
        train_std = np.std(train_scores,axis=1)
        test_mean = np.mean(test_scores,axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_mean, color='blue', marker='o',markersize=5, label='Training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std,train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean,color='green', linestyle='--', marker='s', 
                markersize=5, label='Validation accuracy')

        plt.fill_between(train_sizes,test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.grid()
        plt.xlabel('Number of training examples')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
        plt.savefig('%s/learning_curve.pdf'%path)

    def precision_recall_f1_score(self, best_model):
        # 精确率和召回率 f1分数
        from sklearn.metrics import precision_score, recall_score, f1_score

        Y_pred = best_model.predict(self.X_test)
        pre_val = precision_score(y_true= self.y_test, y_pred=Y_pred, average='macro')
        precision_score_result = 'Precision score: {0:.3f}\n'.format(pre_val)
        report.write(precision_score_result)

        rec_val = recall_score(y_true= self.y_test, y_pred=Y_pred, average='macro')
        recall_score_result = 'Recall score: {0:.3f}\n'.format(rec_val)
        report.write(recall_score_result)

        f1_val = f1_score(y_true= self.y_test, y_pred=Y_pred, average='macro')
        f1_score_result = 'F1_score: {0:.3f}\n'.format(f1_val)
        report.write(f1_score_result)

    def Roc_cruve(self, best_model, path):
        # 绘制ROC曲线
        from sklearn.metrics import roc_curve, auc
        from numpy import interp

        cv = list(self.stratified_kfold.split(self.X_train, self.y_train))

        plt.figure(figsize=(7,5))
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv):
            probas = best_model.fit(self.X_train[train], self.y_train[train]).predict_proba(self.X_train[test])
            fpr, tpr, thresholds = roc_curve(self.y_train[test], probas[:,1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label = f'ROC fold {i+1} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], linestyle= '--', color = (0.6, 0.6, 0.6), label = 'Random guessing (area=0.5)')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--', label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)

        plt.plot([0, 0, 1], [0, 1, 1], linestyle= ':', color = 'black', label = 'Perfect performance (area=1.0)')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
        plt.savefig('%s/ROC_curve.pdf'%path)

    def sha(self, best_model, algorithm, path):
        # 使用shap为模型进行解释
        import shap
        shap.initjs()
        explainer = shap.KernelExplainer(model=best_model.predict, data=X)
        shap_values = explainer.shap_values(X) 
        
        # 生成单个样本的力图
        #for i in range(len(inputData)):
            #shap.force_plot(explainer.expected_value, shap_values[i, :], 
                            #inputData.iloc[i, :-1], show=False, matplotlib=True).savefig('%s_force_%s.png'%(algorithm, i))
        # 生成
        shap.summary_plot(shap_values, features_select_maxrix.iloc[:,:-1])
        plt.savefig('%s/summary_plot.png'%path)
        return shap_values



# ==========================================================================================
#                                         结果输出
# ==========================================================================================
os.chdir(r'/Users/yangpei/YangPei/after_sale/ml/meta')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# lasso筛选特征
if __name__ == "__main__":
    select_algorithm = features_select_algorithm(X_train, X_test, y_train, y_test)
    features, best_alpha = select_algorithm.lasso_algorithm('lasso')
    col_names = [x for x in list(features)]
    col_names.extend(['Label'])
    features_select_maxrix = inputData.loc[:, col_names]
    features_select_maxrix.to_csv('select_features_Lasso.txt', sep='\t', index=False)

    X = features_select_maxrix.iloc[:, 0: -1].values
    y = features_select_maxrix.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# random_froest 筛选特征
if __name__ == "__main__":
    select_algorithm = features_select_algorithm(X_train, X_test, y_train, y_test)
    features = select_algorithm.rfc()
    col_names = [x for x in list(features)]
    col_names.extend(['Label'])
    features_select_maxrix = inputData.loc[:, col_names]
    features_select_maxrix.to_csv('select_features_randomFroest.txt', sep='\t', index=False)
    X = features_select_maxrix.iloc[:, 0: -1].values
    y = features_select_maxrix.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# 支持向量机
report = open('svm/SVM_report.txt', 'a')
if __name__ == "__main__":
    model = ml_algorithm(X_train, X_test, y_train, y_test)
    svm_model, clf = model.svm()
    model.best_params(clf)
    model.test_accurary(svm_model)
    model.cross_val_score(svm_model)
    model.confusion_matrix_est(svm_model, 'svm')
    model.learning_curve_plot(svm_model, 'svm')
    model.precision_recall_f1_score(svm_model)
    model.Roc_cruve(svm_model, 'svm')
    shap_plot = model.sha(svm_model, 'svm', 'svm')
report.close()

"""
# 逻辑回归
report = open('lr/LR_report.txt', 'a')
if __name__ == "__main__":
    model = ml_algorithm(X_train, X_test, y_train, y_test)
    lr_model, clf = model.lr()
    model.best_params(clf)
    model.test_accurary(lr_model)
    model.cross_val_score(lr_model)
    model.confusion_matrix_est(lr_model, 'lr')
    model.learning_curve_plot(lr_model, 'lr')
    model.precision_recall_f1_score(lr_model)
    model.Roc_cruve(lr_model, 'lr')
    shap_plot = model.sha(lr_model, 'lr', 'lr')
report.close()



# 随机森林
report = open('rf/RF_report.txt', 'a')
if __name__ == "__main__":
    model = ml_algorithm(X_train, X_test, y_train, y_test)
    rf_model, clf = model.rf()
    model.best_params(clf)
    model.test_accurary(rf_model)
    model.cross_val_score(rf_model)
    model.confusion_matrix_est(rf_model, 'rf')
    model.learning_curve_plot(rf_model, 'rf')
    model.precision_recall_f1_score(rf_model)
    model.Roc_cruve(rf_model, 'rf')
    shap_plot = model.sha(rf_model, 'rf', 'rf')
report.close()


# xgboost
report = open('xgb/XGB_report.txt', 'a')
if __name__ == "__main__":
    model = ml_algorithm(X_train, X_test, y_train, y_test)
    xgb_model, clf = model.xgb()
    model.best_params(clf)
    model.test_accurary(xgb_model)
    model.cross_val_score(xgb_model)
    model.confusion_matrix_est(xgb_model, 'xgb')
    model.learning_curve_plot(xgb_model, 'xgb')
    model.precision_recall_f1_score(xgb_model)
    model.Roc_cruve(xgb_model, 'xgb')
    shap_plot = model.sha(xgb_model, 'xgb', 'xgb')
report.close()
"""

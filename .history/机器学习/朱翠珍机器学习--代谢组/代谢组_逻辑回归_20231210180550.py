from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir(r'E:\售后\朱翠珍--机器学习\代谢组')
df = pd.read_excel('lasso_select_feature_meta.xlsx')
final_result = open('LogisticRegression_Report.txt', 'a')

# 随机森林不要求假设数据线性可分，因此无需进行归一化，直接划分数据集
x = df.iloc[:, 0:87].values
y = df.iloc[:, -1].values

# ============== step.2 数据集划分 ==============
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y) # stratify表示支持按比例分层策略

# ============== step.3 构建pipline对象 ==============
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 构建pipeline对象
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=1,max_iter=10000, penalty='l1',
                                           multi_class='ovr', solver='liblinear'))

# 设置超参数并使用网络搜索法构建学习器
param_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_grid = [{'logisticregression__C': param_range}]

# 创建分层K折交叉验证对象、创建超参数搜索对象
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, cv=stratified_kfold, n_jobs=3, refit=True)

# 嵌套交叉验证评分器
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=gs, X=X_train, y=Y_train, 
                         scoring='accuracy', cv=5, n_jobs=3)

# 交叉验证评分
LogisticRegression_best_params = 'Best params are {0}\n'.format(gs.best_params_)
final_result.write(LogisticRegression_best_params)

LogisticRegression_cross_val_score = 'CV accuracy: {0:.3f}' f'+/-{1:.3f}\n'.format(np.mean(scores), np.std(scores))
final_result.write(LogisticRegression_cross_val_score)

# 使用测试集估计模型的性能
clf = gs.fit(X=X_train, y=Y_train)
LogisticRegression_Test_accuracy = 'Test accuracy: {0:.3f}\n'.format(clf.score(X_test, Y_test))
final_result.write(LogisticRegression_Test_accuracy)





# 模型评估--学习曲线
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_lr, X=X_train, y=Y_train, 
                                                        train_sizes = np.linspace(0.1, 1.0, 10), 
                                                        cv=stratified_kfold, n_jobs=3)

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
plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.rcParams['figure.figsize'] = (6, 8)
plt.savefig('LogisticRegression_learning_curve.pdf')
plt.show()

# ============== 8.1 混淆矩阵 ============== 
from sklearn.metrics import confusion_matrix
Y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)

# 绘制混淆矩阵图
fig, ax = plt.subplots(figsize = (6, 6))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i,j], va = 'center', ha='center')
ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('LogisticRegression_confusion_matrix.pdf')
plt.show()

# 8.2 精确率和召回率
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

Y_pred = clf.predict(X_test)
pre_val = precision_score(y_true= Y_test, y_pred=Y_pred, average='macro')
LogisticRegression_Precision = 'Precision: {0:.3f}\n'.format(pre_val)
final_result.write(LogisticRegression_Precision)

rec_val = recall_score(y_true= Y_test, y_pred=Y_pred, average='macro')
LogisticRegression_Recall = 'Recall: {0:.3f}\n'.format(rec_val)
final_result.write(LogisticRegression_Recall)
final_result.close()

# 8.3 绘制ROC曲线
from sklearn.metrics import roc_curve, auc
from numpy import interp

cv = list(stratified_kfold.split(X_train, Y_train))

fig = plt.figure(figsize=(6, 8))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = gs.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])
    fpr, tpr, thresholds = roc_curve(Y_train[test], probas[:,1], pos_label=1)
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
plt.savefig('LogisticRegression_ROC_curve.pdf')
plt.show()



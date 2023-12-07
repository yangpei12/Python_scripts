import pandas as pd
import numpy as np

# 加载威斯康星数据集合
df = pd.read_csv(r'', header=None)

# ============== step.1 数据转换 ==============
# 将30个特征存入Numpy数组x中(直接调用df.values属性即可即可)。
# 使用labelEncoder对象将类别标签从原始的字符串转换为整数
from sklearn.preprocessing import LabelEncoder
x = df.loc[:, 2:].values
y = df.loc[:, 1].values
label_encode = LabelEncoder() # 创建LabelEncoder对象
y = label_encode.fit_transform(y)

# ============== step.2 数据集划分 ==============
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y) # stratify表示支持按比例分层策略


# ============== step.3 数据标准化 ==============
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 数据标准化及维度压缩
pipe_line_regression = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())

# 拟合数据构建模型
pipe_line_regression.fit(X_train, Y_train)

# 利用模型进行预测
y_pre = pipe_line_regression.predict(X_test)

test_acc = pipe_line_regression.score(X_test, Y_test)
print(f'Test accurary:{test_acc:.3f}')

# ============== step.4.1 交叉验证--分层交叉验证 ==============
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10).split(X=X_train, y= Y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_line_regression.fit(X_train[train], Y_train[train])
    score = pipe_line_regression.score(X_train[test], Y_train[test])
    scores.append(score)
    print(f'fold:{k+1:02d}, ' f'class distr.:{np.bincount(Y_train[train])}, ' f'acc : {score:.3f}')

mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f'\nCV accuracy: {mean_acc: .3f} +/- {std_acc:.3f}')

# ============== step.4.2 交叉验证--分层交叉验证简洁版 ==============
# 创建分层K折交叉验证对象
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# 交叉验证评分器
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_line_regression, X=X_train, y=Y_train, cv=stratified_kfold, n_jobs=1)


# ============== step.5 绘制学习曲线 ==============
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# penalty='l2' 使用l2正则化，对特征进行惩罚
pipe_line_regression = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=10000))

# train_sizes参数用于控制用于生成学习曲线的训练样本的绝对数量或相对数量，训练样本中共计405个样本，
# np.linspace(0.1, 1.0, 10) = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
# 405 * 0.1 = 40个训练样本、405*0.2 = 81个样本，405*0.3 = 122个样本, ...
# 根据训练样本数量的不断增加，模型在训练和测试集上的的学习曲线会不断趋同
train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_line_regression, X=X_train, y=Y_train, 
                                                        train_sizes = np.linspace(0.1, 1.0, 10), 
                                                        cv=10, n_jobs=1)

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

# 输出学习曲线图
plt.show()

# ============== step.6 绘制验证曲线 ==============
# 使用验证曲线评估模型的过拟合和欠拟合的问题
# 验证曲线是关于模型参数和模型准确率的函数，因此横坐标是模型的参数，纵坐标是准确率
from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_line_regression, X=X_train, y=Y_train, 
                                             param_name= 'logisticregression__C',
                                             param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color = 'blue', 
         marker='o', markersize=5, label = 'Tranining Accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(param_range, train_mean, color='green', linestyle='--', 
         marker='s', markersize=5, label='Validation Accuracy')

plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.00])

# 输出验证曲线图
plt.show()

# ============== step.7 超参数配置 ==============
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# ============== step.7.1 网络搜索法 ==============
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# 将GridSearchCV的param_grid参数设置为字典列表，以指定想要调整的参数。对于线性SVM只评估正则化参数C；
# 对于RBF，调整svc__C和svc__gamma参数。
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},

              {'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]

# 执行10折交叉验证，设置scoring='accuracy' 计算10折的平均准确率
grid_search = GridSearchCV(estimator= pipe_svc, param_grid=param_grid, scoring='accuracy', 
                           cv = 10, refit=True, n_jobs=-1)


grid_search = grid_search.fit(X=X_train, y=Y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)

# 选择最佳模型，然后通过测试集评估模型的性能
clf = grid_search.best_estimator_
# clf.fit(X_train, Y_train)
print(f'Test accuracy: {clf.score(X_test, Y_test):.3f}')

# 注意，在完成网格搜索后，不需要通过clf.fit(X_train, Y_train)在训练数据集上手动拟合具有最佳设置（grid_search.best_estimator_）的模型。
# GridSearchCV类有一个refit参数，如果设置refit=True(默认)，它将自动在整个训练数据集上重新拟合gs.best_estimator

# ============== step.7.2 随机搜索法 ==============
# 随机搜索从分布（或离散集）中随机抽取超参数。



# ============== step.7.3 连续减半搜索法 ==============
# ============== step.7.4 嵌套交叉验证 ==============

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring= 'accuracy', cv=2)

scores = cross_val_score(gs, X_train, Y_train, scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f}' f'+/-{np.std(scores):.3f}')

# ============== step.8 模型性能评估 ==============
# ============== 8.1 混淆矩阵 ============== 
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X=X_train, y=Y_train)
Y_pred = pipe_svc.predict(X_test)
conf_matrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
# 绘制混淆矩阵图
fig, ax = plt.subplots(figsize = (2.5, 2.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i,j], va = 'center', ha='center')
ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# 8.2 精确率和召回率
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import matthews_corrcoef

pre_val = precision_score(y_true= Y_test, y_pred=Y_pred)
print(f'Precision: {pre_val:.3f}')

rec_val = recall_score(y_true= Y_test, y_pred=Y_pred)
print(f'Recall: {rec_val:.3f}')

f1_val = f1_score(y_true= Y_test, y_pred=Y_pred)
print(f'F1 Score: {f1_val:.3f}')

mcc_val = matthews_corrcoef(y_true= Y_test, y_pred=Y_pred)
print(f'MCC Score: {mcc_val:.3f}')

# 8.2.1 使用make_scorer在GridSearchCV中传递score参数
from sklearn.metrics import make_scorer
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C': c_gamma_range, 'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range, 'svc__kernel': ['rbf'], 'svc__gamma': c_gamma_range}]

scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring= scorer, cv=10, refit=True)

# refit参数为True，因此只用将gs对象进行fit就行（默认使用最佳的学习器拟合），不用把最佳学习器重新拟合
gs = gs.fit(X_train, Y_train)

print(f'Best score: {gs.best_score_}')
print(f'Best estimator: {gs.best_estimator_}')

# 8.3 绘制ROC曲线
from sklearn.metrics import roc_curve, auc
from numpy import interp

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(penalty='l2', random_state=1, solver='lbgfs', C=100))

X_train2 = X_train[:, [4,14]]
cv = list(StratifiedKFold(n_splits=3).split(X_train, Y_train))

fig = plt.figure(figsize=(7,5))
mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], Y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(Y_train[test], probas[:,1], pos_label=1)
    mean_tpr[0] = 0.0
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
plt.legend(loc= 'lower right')
plt.show()




# 8.4 多分类器评价指标

import pandas as pd
import numpy as np

# 加载威斯康星数据集合
df = pd.read_csv(r'E:\售后\机器学习\dataset\breast+cancer+wisconsin+diagnostic\wdbc.data', header=None)

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
pipe_line_regression = make_pipeline(StandardScaler(), LogisticRegression(penalty='12', max_iter=10000))
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_line_regression, X=X_train, y=Y_train, train_sizes=np.linspace(0.1, 1.0, 10),cv=10, n_jobs=1)

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

# ============== step.6 绘制验证曲线 ==============
from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_line_regression, X=X_train, y=Y_train, 
                                             param_name= 'logisticregression__C',
                                             param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color = 'blue', maker='o', makersize=5, label = 'Tranining Accuracy')
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
plt.show()

# ============== step.7 超参数配置 ==============
# ============== step.7.1 网络搜索法 ==============
# ============== step.7.2 随机搜索法 ==============
# ============== step.7.3 连续减半搜索法 ==============
# ============== step.7.4 嵌套交叉验证 ==============
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring= 'accuracy', cv=2)

scores = cross_val_score(gs, X_train, Y_train, scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f}' f'+/-{np.std(scores):.3f}')

# ============== step.8 模型性能评估 ==============

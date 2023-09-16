import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
import os

# 设置工作路径
os.chdir(r'E:\售后\机器学习\xgboost\Input')

# 使用pandas将数据读取为数据框
input_data_frame = pd.read_excel('xgboost_input_data.xlsx')
data_feature =  input_data_frame.iloc[:,0:-2]
data_label = input_data_frame.iloc[:,-1]

# 将数据集划分为测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(data_feature, data_label, test_size=0.3, random_state=23)

# 执行网络搜索
model = XGBClassifier(gamma=0)


param_grid = {'learning_rate': [0.01], 'n_estimators': [300,400,500], 'max_depth': [5,10,15], 'min_child_weight': [1]}



kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=2, cv=kfold, verbose=1, return_train_score=True, refit=True)
grid_result = grid_search.fit(X_train, y_train)

# 判断模型准确性
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
bclf = grid_result.best_estimator_
y_true = y_test
y_pred = bclf.predict(X_test)
y_pred_pro = bclf.predict_proba(X_test)

y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
print(classification_report(y_true, y_pred))
auc_value = roc_auc_score(y_true, y_scores)

#绘制ROC曲线
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

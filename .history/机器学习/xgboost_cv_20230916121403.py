import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
import os

# 设置工作路径
os.chdir(r'E:\售后\机器学习\xgboost\Input')

# 使用pandas将数据读取为数据框
input_data_frame = pd.read_excel('xgboost_input_data.xlsx')
data_feature =  input_data_frame.iloc[:,0:-2]
data_label = input_data_frame.iloc[:,-1]

# 构建分类器
clf = XGBClassifier(learning_rate = 0.1, 
                    n_estimators = 400,
                    max_depth = 10,
                    min_child_weight = 1,
                    min_split_loss = 0,
                    booster = 'gbtree',
                    random_state = 23)
# 重采样策略
repcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

# 执行交叉验证
xgb_cv = cross_validate(estimator=clf, X=data_feature, y=data_label,
                         scoring='roc_auc', cv=repcv, return_estimator=True)

# 导出每次迭代的重要性
for idx,estimator in enumerate(xgb_cv['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                        columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)


# 判断模型准确性
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def estimator_evaluate(n_estimator):
    bclf = xgb_cv['estimator'][n_estimator]
    # 将数据集划分为测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_label, test_size=0.2, random_state=23)
    bclf.fit(X_train, y_train)
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

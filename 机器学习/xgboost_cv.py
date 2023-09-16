import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
import os

# 设置工作路径
os.chdir(r'/Users/yangpei/YangPei/File/Python/Input')

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


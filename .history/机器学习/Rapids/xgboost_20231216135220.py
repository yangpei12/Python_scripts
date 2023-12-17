# 数据整理
import cudf
import os

# 划分数据集
os.chdir(r'/mnt/d/售后/朱翠珍--机器学习/转录组')
mRNA_data = cudf.read_csv('lasso_select_feature_mRNA.csv')

a = mRNA_data.iloc[:, 0:-1].values
b = mRNA_data.iloc[:, -1].values

from cuml.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( a, b, random_state = 1 )

# 构建学习器

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

param_grid = [{'xgbclassifier__learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05], 
               'xgbclassifier__max_depth': [5,6,7,8,9,10], 
               'xgbclassifier__min_child_weight': [1],
               'xgbclassifier__booster': ['gbtree'],
               'xgbclassifier__n_estimators':[200, 300, 400],
               'xgbclassifier__reg_alpha':[5, 6, 7, 8],
               'xgbclassifier__tree_method':'gpu_hist'}]

import xgboost
xgb_model = xgboost.XGBClassifier(num_class=10,
                          metric='multiclass',
                          eval_metric='mlogloss',
                          random_state=1,
                          n_jobs=0,
                          use_label_encoder= False)

# nested cross validation
inner_cv = StratifiedKFold(n_splits=4, shuffle=True , random_state=1)
outer_cv = StratifiedKFold(n_splits=4, shuffle=True , random_state=1)

# inner_cv
gcv = GridSearchCV(xgb_model, 
                   param_grid, 
                   scoring='balanced_accuracy', 
                   cv=inner_cv,
                   n_jobs=1, 
                   return_train_score=False)

# outer cv
results = cross_validate(gcv,
                         X,
                         y,
                         scoring='balanced_accuracy',
                         cv=outer_cv,
                         n_jobs=2, 
                         return_train_score=False)

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
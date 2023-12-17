# 数据整理
import cudf
import os

# 划分数据集
os.chdir(r'/mnt/d/售后/朱翠珍--机器学习/转录组')
mRNA_data = cudf.read_csv('lasso_select_feature_mRNA.csv')

from cuml.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state = 0 )

# 构建学习器
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

params = {
  "colsample_bynode": 0.8,
  "learning_rate": 1,
  "max_depth": 5,
  "num_parallel_tree": 100,
  "objective": "binary:logistic",
  "subsample": 0.8,
  "tree_method": "hist",
  "device": "cuda",
}

xgb_model = XGBClassifier(num_class=10,
                          metric='multiclass',
                          eval_metric='mlogloss',
                          random_state=911,
                          tree_method='gpu_hist',
                          n_jobs=0,
                          use_label_encoder= False)

# nested cross validation
inner_cv = StratifiedKFold(n_splits=4, shuffle=True ,random_state=123)
outer_cv = StratifiedKFold(n_splits=4, shuffle=True ,random_state=321)

# inner_cv
gcv = GridSearchCV(xgb_model, 
                   params, 
                   scoring='balanced_accuracy', 
                   cv=inner_cv,
                   iid=False,
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
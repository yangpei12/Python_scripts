# 数据整理
import pandas as pd
import os
import numpy as np
import xgboost
# 划分数据集
os.chdir(r'/mnt/d/售后/朱翠珍--机器学习/转录组')
mRNA_data = pd.read_csv('lasso_select_feature_mRNA.csv')

feature = mRNA_data.iloc[:, 0:-1].values
target = mRNA_data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( feature, target, random_state = 1 )

# 构建学习器
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

dtrain = xgboost.DMatrix(X_train, label=y_train)
dvalidation = xgboost.DMatrix(X_test, label=y_test)

params = {'tree_method': 'gpu_hist', 'eval_metric': 'auc', 'objective': 'binary:logistic'}

evallist = [(dvalidation, 'validation'), (dtrain, 'train')]
num_round = 10

xgboost.train(params, dtrain, num_round, evallist)




"""
xgb_model = XGBClassifier(num_class=3,
                          metric='multiclass',
                          eval_metric='mlogloss',
                          random_state=1,
                          tree_method="hist", 
                          device="cuda")

# nested cross validation
inner_cv = StratifiedKFold(n_splits=4, shuffle=True , random_state=1)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True , random_state=1)

# inner_cv
gcv = GridSearchCV(xgb_model, 
                   param_grid, 
                   scoring='balanced_accuracy', 
                   cv=inner_cv,
                   n_jobs=1, 
                   return_train_score=False)

# outer cv
results = cross_validate(gcv,
                         feature,
                         target,
                         scoring='balanced_accuracy',
                         cv=outer_cv,
                         n_jobs=2, 
                         return_train_score=False)
"""
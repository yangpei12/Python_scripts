import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import os

# 设置工作路径
os.chdir(r'E:\售后\机器学习\xgboost')

# 使用pandas将数据读取为数据框
input_data_frame = pd.read_excel('xgboost_input_data.xlsx')
data_feature =  input_data_frame.iloc[:,0:-2]
data_label = input_data_frame.iloc[:,-1]


# grid search
model = XGBClassifier(gamma=0)


param_grid = {'learning_rate': 0.3, 'n_estimators': 500, 'max_depth': 15, 'min_child_weight': 1}



kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=23)

grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(data_feature, data_label)

import pandas as pd
import numpy as np
import xgboost as xgb
import os

# 设置工作路径
os.chdir(r'E:\售后\机器学习\xgboost')

# 读取数据
input_data = pd.read_excel('Input/xgboost_input_data.xlsx')

# 
dtrain = xgb.DMatrix(os.path.join(CURRENT_DIR, '../data/agaricus.txt.train'))
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}
num_round = 2

print('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed=0,
       callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)])

print('running cross validation, disable standard deviation display')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value
res = xgb.cv(param, dtrain, num_boost_round=10, nfold=5,
             metrics={'error'}, seed=0,
             callbacks=[xgb.callback.EvaluationMonitor(show_stdv=False),
                        xgb.callback.EarlyStopping(3)])
print(res)
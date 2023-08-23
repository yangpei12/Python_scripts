# 导入工具库
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 设置工作目录
os.chdir(r'E:\scripts_input_R_python\python\Input')
# 用pandas读入数据
data = pd.read_csv('pima-indians-diabetes.csv')
 
# 做数据切分
train, test = train_test_split(data)
 
# 转换成Dmatrix格式
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target_column = 'Outcome'
 
# 取出Dataframe的numpy数组值去初始化DMatrix对象
xgtrain = xgb.DMatrix(train[feature_columns].values, train[target_column].values)
xgtest = xgb.DMatrix(test[feature_columns].values, test[target_column].values)
 
#参数设定
param = {'max_depth':5, 'eta':0.1, 'silent':1, 'subsample':0.7, 'colsample_bytree':0.7, 'objective':'binary:logistic' }
 
# 设定watchlist用于查看模型状态
watchlist  = [(xgtest,'eval'), (xgtrain,'train')]
num_round = 10
bst = xgb.train(param, xgtrain, num_round, watchlist)
 
# 使用模型预测
preds = bst.predict(xgtest)
 
# 判断准确率
labels = xgtest.get_label()
print('错误类为%f' % \
       (sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
 
# 模型存储
bst.save_model('xgboost_practise.model')
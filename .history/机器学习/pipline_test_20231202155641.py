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






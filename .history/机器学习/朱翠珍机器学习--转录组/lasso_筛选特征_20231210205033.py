# lasso消除
# 通过交叉验证来筛选最佳的alpha值
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
os.chdir(r'E:\售后\朱翠珍--机器学习\转录组')
df = pd.read_excel('finnal_ml_input.xlsx')
x = df.iloc[:, 0:13023].values
y = df.iloc[:, -1].values

ss = StandardScaler()
std_data = ss.fit_transform(x)

lasso_cv = LassoCV(cv=10)
lasso_cv.fit(std_data, y)
optimal_alpha = lasso_cv.alpha_

# 选择最佳的alpha值筛选特征
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
lasso = Lasso(alpha=optimal_alpha)  # 调整alpha参数以控制特征选择的严格程度
sfm = SelectFromModel(lasso, max_features=1000)
sfm.fit(std_data, y)  # X是特征数据，y是目标变量

col_index = [i for i, value in enumerate(list(sfm.get_support())) if value == True]
selected_features = df.columns[col_index]  # 获取选定的特征


lasso_select = df.iloc[:, col_index + [-1]]
lasso_select.to_excel( 'lasso_select_feature_mRNA.xlsx', index=False)


# 绘制lasso路径图
import matplotlib.pyplot as plt

alphas = lasso_cv.alphas_
mse = np.mean(lasso_cv.mse_path_, axis=1)

plt.plot(np.log10(alphas), mse)
plt.xlabel('log10(alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-validated Lasso paths')
plt.savefig('Lasso_mse_path.pdf')
plt.show()


"""
# 递归消除
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 100  # Minimum number of features to consider
clf = LogisticRegression()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=3,
)
rfecv.fit(x, y)

print(f"Optimal number of features: {rfecv.n_features_}")
"""
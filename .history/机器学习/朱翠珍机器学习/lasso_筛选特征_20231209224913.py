# lasso消除
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
lasso = Lasso(alpha=11)  # 调整alpha参数以控制特征选择的严格程度
sfm = SelectFromModel(lasso, max_features=1000)
sfm.fit(x, y)  # X是特征数据，y是目标变量


col_index = [i for i, value in enumerate(list(sfm.get_support())) if value == True]

selected_features = df.columns[col_index]  # 获取选定的特征

from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=10)
lasso_cv.fit(x, y)
optimal_alpha = lasso_cv.alpha_


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
# data
from scipy.sparse import data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
# eval
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


raw_boston = datasets.load_boston()

X = raw_boston.data
y = raw_boston.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 42) 

std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

clf_lr = LinearRegression()
clf_lr.fit(X_tn_std, y_tn)
print('clf_lr.coef_ : ', clf_lr.coef_)
print('clf_lr.intercept_ : ', clf_lr.intercept_)

clf_ridge = Ridge(alpha = 1)
clf_ridge.fit(X_tn_std, y_tn)
print('clf_ridge.coef_ : ', clf_ridge.coef_)
print('clf_ridge.intercept_ : ', clf_ridge.intercept_)

clf_lasso = Lasso(alpha = 0.01)
clf_lasso.fit(X_tn_std, y_tn)
print('clf_lasso.coef_ : ', clf_lasso.coef_)
print('clf_lasso.intercept_ : ', clf_lasso.intercept_)

clf_elastic = ElasticNet(alpha = 0.01, l1_ratio = 0.01)
clf_elastic.fit(X_tn_std, y_tn)
print('clf_elastic.coef_ : ', clf_elastic.coef_)
print('clf_elastic.intercept_ : ', clf_elastic.intercept_)

pred_lr = clf_lr.predict(X_te_std)
pred_ridge = clf_ridge.predict(X_te_std)
pred_lasso = clf_lasso.predict(X_te_std)
pred_elastic = clf_elastic.predict(X_te_std)

print('linear reg R2 : ', r2_score(y_te, pred_lr))
print('ridge reg R2 : ', r2_score(y_te, pred_ridge))
print('lasso reg R2 : ', r2_score(y_te, pred_lasso))
print('elastic reg R2 : ', r2_score(y_te, pred_elastic))

print('linear reg MSE : ', mean_squared_error(y_te, pred_lr))
print('ridge reg MSE : ', mean_squared_error(y_te, pred_ridge))
print('lasso reg MSE : ', mean_squared_error(y_te, pred_lasso))
print('elastic reg MSE : ', mean_squared_error(y_te, pred_elastic))
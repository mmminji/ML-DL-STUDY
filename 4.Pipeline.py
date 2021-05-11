from Data import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = boston()
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2, random_state=123)


# == not using pipeline ==
std = StandardScaler()
x_train_std = std.fit_transform(X_train)   # train : fit --> transform
x_test_std = std.transform(X_test)         # test : only transform

clf = LinearRegression()
clf.fit(x_train_std, Y_train)
pred = clf.predict(x_test_std)

print(mean_squared_error(Y_test, pred))
# 28.19248575846957


# == using pipeline ==
std_linear = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regression', LinearRegression())
])

std_linear.fit(X_train, Y_train)
pred_pipe = std_linear.predict(X_test)

print(mean_squared_error(Y_test, pred_pipe))
# 28.19248575846957  --> same result
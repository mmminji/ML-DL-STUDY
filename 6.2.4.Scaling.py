from sklearn.preprocessing import StandardScaler
std = StandardScaler()

from sklearn.preprocessing import RobustScaler
robust = RobustScaler()

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

from sklearn.preprocessing import Normalizer
normal = Normalizer()

from Data import *
from sklearn.model_selection import train_test_split
data = boston()

# == fit & transform == 
std.fit(data[['TAX']])
TAX_std1 = std.transform(data[['TAX']])

# == fit & transform at once ==
TAX_std2 = std.fit_transform(data[['TAX']])

# == fit train & transform test ==
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2, random_state=123)
x_train_std = minmax.fit_transform(X_train)
x_test_sd = minmax.transform(X_test)
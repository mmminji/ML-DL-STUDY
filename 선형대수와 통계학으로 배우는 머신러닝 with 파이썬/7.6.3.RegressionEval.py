y_pred = [2.5, 0.0, 2, 8]
y_true = [3, -0.5, 2, 7]

# == MAE ==
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_true, y_pred))
# 0.5

# == MSE ==
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_true, y_pred))
# 0.375

# == R2 score ==
from sklearn.metrics import r2_score
print(r2_score(y_true, y_pred))
# 0.9486081370449679
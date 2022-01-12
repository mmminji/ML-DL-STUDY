from numpy.core.numeric import cross
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# == Data load == 
raw_wine = datasets.load_wine()
X = raw_wine.data
y = raw_wine.target

# == Train & Test split == 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# == Data Scaling == 
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_sd = std.transform(X_test)

# == Grid Search Training == 
param_grid = {'kernel' : ('linear', 'rbf'),
                'C' : [0.5, 1, 10, 100]}
svc = svm.SVC(random_state=123)
grid_cv = GridSearchCV(svc, param_grid, cv = 5, scoring = 'accuracy')
grid_cv.fit(X_train_std, y_train)

# == Grid Search Results == 
grid_cv.cv_results_
grid_cv.best_params_
# {'C': 10, 'kernel': 'rbf'}
clf = grid_cv.best_estimator_

# == Cross Validation Score == 
metrics = ['accuracy', 'f1_macro']
cv_scores = cross_validate(clf, X_train_std, y_train, cv = 5, scoring = metrics)
# {'fit_time': array([0.00107574, 0.00099707, 0.00099659, 0.00099826, 0.00099754]),
#  'score_time': array([0.00703573, 0.00101686, 0.00099754, 0.00099683, 0.0009973 ]),
#  'test_accuracy': array([0.96551724, 1.        , 0.96428571, 1.        , 0.96428571]),
#  'test_f1_macro': array([0.96102564, 1.        , 0.95636364, 1.        , 0.95986622])}
print(cv_scores['test_accuracy'].mean())
# 0.9788177339901478
print(cv_scores['test_accuracy'].std())
# 0.017301092934813727

# == Predict & Accuracy == 
pred_svm = clf.predict(X_test_sd)
acc = accuracy_score(y_test, pred_svm)
print(acc)
# 0.9722222222222222

# == Confusion Matrix == 
conf_matrix = confusion_matrix(y_test, pred_svm)
print(conf_matrix)
# [[ 8  0  0]
#  [ 0 11  0]
#  [ 0  1 16]]

# == Classification Report == 
class_report = classification_report(y_test, pred_svm)
print(class_report)
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         8
#            1       0.92      1.00      0.96        11
#            2       1.00      0.94      0.97        17

#     accuracy                           0.97        36
#    macro avg       0.97      0.98      0.98        36
# weighted avg       0.97      0.97      0.97        36
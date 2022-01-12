import pandas as pd
# data
from sklearn import datasets
from sklearn import metrics
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
# eval
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


raw_wine = datasets.load_wine()

X = raw_wine.data
y = raw_wine.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 42) 

std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

param_grid = {'kernel' : ('linear', 'rbf'),
                'C' : [0.5, 1, 10, 100]}
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)  # 클래스 비율을 유지하면서 데이터 추출
svc = svm.SVC(random_state = 42)
grid_cv = GridSearchCV(svc, param_grid, cv = kfold, scoring = 'accuracy')
grid_cv.fit(X_tn_std, y_tn)

# pd.DataFrame(grid_cv.cv_results_).T
print('best_score : ', grid_cv.best_score_)
print('best_params : ', grid_cv.best_params_)
best_clf = grid_cv.best_estimator_
print(best_clf)

metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_score = cross_validate(best_clf, X_tn_std, y_tn, cv = kfold, scoring = metrics)
# pd.DataFrame(cv_score).T

svm_pred = best_clf.predict(X_te_std)
print('prediction : ', svm_pred)

acc = accuracy_score(y_te, svm_pred) 
print('accuracy : ', acc)

conf_matrix = confusion_matrix(y_te, svm_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, svm_pred)
print(class_report)
# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.linear_model import LogisticRegression
# eval
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


raw_cancer = datasets.load_breast_cancer()

X = raw_cancer.data
y = raw_cancer.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 42) 

std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

clf_logi_l2 = LogisticRegression(penalty = 'l2') # l1, l2, elasticnet, none
clf_logi_l2.fit(X_tn_std, y_tn)

print('clf_logi_l2.coef_ : ', clf_logi_l2.coef_)
print('clf_logi_l2.intercept_ : ', clf_logi_l2.intercept_)

logi_l2_pred = clf_logi_l2.predict(X_te_std)
print('prediction : ', logi_l2_pred)

logi_l2_pred_prob = clf_logi_l2.predict_proba(X_te_std)
print('prediction_probability : ', logi_l2_pred_prob)

precision = precision_score(y_te, logi_l2_pred)
print('precision : ', precision)

conf_matrix = confusion_matrix(y_te, logi_l2_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, logi_l2_pred)
print(class_report)
# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
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

clf_bagging = BaggingClassifier(base_estimator = GaussianNB(), n_estimators = 10, random_state = 42)
clf_bagging.fit(X_tn_std, y_tn)

bagging_pred = clf_bagging.predict(X_te_std)

acc = accuracy_score(y_te, bagging_pred)
print('acc : ', acc)

conf_matrix = confusion_matrix(y_te, bagging_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, bagging_pred)
print(class_report)
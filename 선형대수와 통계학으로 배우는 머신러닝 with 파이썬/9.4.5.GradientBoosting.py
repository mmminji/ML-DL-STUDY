# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.ensemble import GradientBoostingClassifier
# eval
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


raw_cancer = datasets.load_breast_cancer()

X = raw_cancer.data
y = raw_cancer.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 42) 

std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

clf_gb = GradientBoostingClassifier(max_depth = 2, learning_rate = 0.01, random_state = 42) 
clf_gb.fit(X_tn_std, y_tn)

gb_pred = clf_gb.predict(X_te_std)

acc = accuracy_score(y_te, gb_pred)
print('acc : ', acc)

conf_matrix = confusion_matrix(y_te, gb_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, gb_pred)
print(class_report)
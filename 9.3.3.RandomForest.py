# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.ensemble import RandomForestClassifier
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

clf_rf = RandomForestClassifier(max_depth = 2, random_state = 42)
clf_rf.fit(X_tn_std, y_tn)

rf_pred = clf_rf.predict(X_te_std)

acc = accuracy_score(y_te, rf_pred)
print('acc : ', acc)

conf_matrix = confusion_matrix(y_te, rf_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, rf_pred)
print(class_report)
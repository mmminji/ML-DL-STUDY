# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.neighbors import KNeighborsClassifier
# eval
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


raw_iris = datasets.load_iris()

X = raw_iris.data
y = raw_iris.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 42) 

std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

clf_knn = KNeighborsClassifier(n_neighbors = 2)
clf_knn.fit(X_tn_std, y_tn)

knn_pred = clf_knn.predict(X_te_std)

acc = accuracy_score(y_te, knn_pred)
print('acc : ', acc)

conf_matrix = confusion_matrix(y_te, knn_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, knn_pred)
print(class_report)
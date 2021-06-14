# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn import tree
# eval
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


raw_wine = datasets.load_wine()

X = raw_wine.data
y = raw_wine.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 42) 

std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

clf_tree = tree.DecisionTreeClassifier(random_state = 42)
clf_tree.fit(X_tn_std, y_tn)

tree_pred = clf_tree.predict(X_te_std)
print('prediction : ', tree_pred)

f1 = f1_score(y_te, tree_pred, average = 'macro') # None, 'micro', 'macro', 'weighted'
print('f1 : ', f1)

conf_matrix = confusion_matrix(y_te, tree_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, tree_pred)
print(class_report)
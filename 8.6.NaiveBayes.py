# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.naive_bayes import GaussianNB
# eval
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


raw_wine = datasets.load_wine()

X = raw_wine.data
y = raw_wine.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 42) 

std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

clf_gnb = GaussianNB()
clf_gnb.fit(X_tn_std, y_tn)

gnb_pred = clf_gnb.predict(X_te_std)
print('prediction : ', gnb_pred)

recall = recall_score(y_te, gnb_pred, average = 'macro') # None, 'micro', 'macro', 'weighted'
print('recall : ', recall)

conf_matrix = confusion_matrix(y_te, gnb_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, gnb_pred)
print(class_report)
# data
from sklearn import datasets
# preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# model
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
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

clf1 = LogisticRegression(multi_class = 'multinomial', random_state = 42)
clf2 = svm.SVC(kernel = 'linear', random_state = 42)
clf3 = GaussianNB()
clf_voting = VotingClassifier(estimators = [('lr', clf1), ('svm', clf2), ('gnb', clf3)],
                                voting = 'hard',   # 'hard' : 과반수, 'soft' : 가장 높은 확률
                                weights = [1,1,1])
clf_voting.fit(X_tn_std, y_tn)

voting_pred = clf_voting.predict(X_te_std)

acc = accuracy_score(y_te, voting_pred)
print('acc : ', acc)

conf_matrix = confusion_matrix(y_te, voting_pred)
print('confusion matrix : ', conf_matrix)

class_report = classification_report(y_te, voting_pred)
print(class_report)
# == Accuracy ==
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))
# 0.5

# == Confusion Matrix ==
from sklearn.metrics import confusion_matrix
y_pred = [0, 0, 2, 2, 0, 2]
y_true = [2, 0, 2, 2, 0, 1]
confusion_matrix(y_true, y_pred)
# array([[2, 0, 0],
#        [0, 0, 1],
#        [1, 0, 2]], dtype=int64)

# == Classification Report ==
from sklearn.metrics import classification_report
y_pred = [0, 0, 2, 1, 0]
y_true = [0, 1, 2, 2, 0]
target_name = ['class0', 'class1', 'class2']
print(classification_report(y_true, y_pred, target_names=target_name))
#               precision    recall  f1-score   support

#       class0       0.67      1.00      0.80         2
#       class1       0.00      0.00      0.00         1
#       class2       1.00      0.50      0.67         2

#     accuracy                           0.60         5
#    macro avg       0.56      0.50      0.49         5
# weighted avg       0.67      0.60      0.59         5
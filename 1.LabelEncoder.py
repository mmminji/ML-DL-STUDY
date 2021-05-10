from Data import *
from sklearn.preprocessing import LabelEncoder

data = example_chapter6()
class_label = LabelEncoder()

data_value = data['target'].values
y_new = class_label.fit_transform(data_value)
y_new
# array([1, 0, 2, 3])
data['target'] = y_new

y_ori = class_label.inverse_transform(y_new)
y_ori
# array(['class2', 'class1', 'class3', 'class4'], dtype=object)
data['target'] = y_ori
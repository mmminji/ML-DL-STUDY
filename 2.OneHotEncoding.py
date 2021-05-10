from Data import *
import pandas as pd

# == dataframe type ==
data = example_chapter6()
data['target'] = data['target'].astype(str)
new_data = pd.get_dummies(data['target'], drop_first=True)
#   class1	class2	class3	class4
# 0	    0	    1	    0   	0
# 1 	1   	0	    0   	0
# 2 	0	    0	    1	    0
# 3	    0   	0	    0	    1

new_data2 = pd.get_dummies(data)  # for all columns


# == array type ==
from sklearn.preprocessing import OneHotEncoder

hot_encoer = OneHotEncoder()
y = data[['target']]
y_hot = hot_encoer.fit_transform(y)
y_hot.toarray() 
# array([[0., 1., 0., 0.],
#        [1., 0., 0., 0.],
#        [0., 0., 1., 0.],
#        [0., 0., 0., 1.]])
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)


raw_boston = datasets.load_boston()

X = raw_boston.data
y = raw_boston.target
print(X.shape)
print(set(y))

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 0) 

n_feat = X_tn.shape[1]
epo = 30

model = Sequential()
model.add(Dense(20, input_dim = n_feat, activation = 'relu'))  # output_dim, input_dim, activation(한번에)
model.add(Dense(1))    # output_dim(회귀), 활성화 함수 없으면 선형 함수

model.summary()

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mean_squared_error'])  # 이진형 : binary_crossentropy

hist = model.fit(X_tn, y_tn, epochs = epo, batch_size = 5)

print(model.evaluate(X_tn, y_tn)[1])
print(model.evaluate(X_te, y_te)[1])

epoch = np.arange(1, epo+1)
mse = hist.history['mean_squared_error']
loss = hist.history['loss']

# accuracy graph
plt.plot(epoch, mse, label = 'accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# loss graph
plt.plot(epoch, loss, 'r', label = 'loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# predict dataframe
pred_y = model.predict(X_te).flatten()
res_df = pd.DataFrame(pred_y, columns = ['predict_val'])
res_df['real_val'] = y_te
df_sort = res_df.sort_values(['predict_val'], ascending = True)

idx = np.arange(1, len(df_sort)+1)
plt.scatter(idx, df_sort['real_val'], marker = 'o', label = 'real_val')
plt.plot(idx, df_sort['predict_val'], color = 'r', label = 'predict_val')
plt.xlabel('index')
plt.ylabel('value')
plt.legend()
plt.show()
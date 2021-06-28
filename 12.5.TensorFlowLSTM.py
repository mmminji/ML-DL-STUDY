import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D

np.random.seed(0)
tf.random.set_seed(0)

# data load
(X_tn0, y_tn0), (X_te0, y_test) = imdb.load_data(num_words = 2000)   # 주로 사용하는 단어만 사용
print(X_tn0.shape)     #(25000,)
print(y_tn0.shape)     #(25000,)
print(X_te0.shape)     #(25000,)
print(y_test.shape)    #(25000,)

X_train = X_tn0[:20000]
y_train = y_tn0[:20000]
X_valid = X_tn0[20000:25000]
y_valid = y_tn0[20000:25000]
print(X_train[0])
print(len(X_train[0]))  #218
print(len(X_train[1]))  #189
print(set(y_test))
print(len(set(y_test))) #2

# pre-processing
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_valid = sequence.pad_sequences(X_valid, maxlen=100)
X_test = sequence.pad_sequences(X_te0, maxlen=100)
print(X_train.shape)    #(20000, 100)
print(X_valid.shape)    #(5000, 100)
print(X_test.shape)     #(25000, 100)

# model
model = Sequential()
model.add(Embedding(input_dim = 2000, output_dim = 100))  # input_dim : 총 단어 개수(num_words), output_dim : 데이터 길이(maxlen)
model.add(Conv1D(50, kernel_size = 3,
                    padding = 'valid',
                    activation = 'relu'))
model.add(MaxPooling1D(pool_size = 3))
model.add(LSTM(100, activation = 'tanh'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

# model compile
model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

# train
hist = model.fit(X_train, y_train,
                    epochs = 10, batch_size = 100,
                    validation_data = (X_valid, y_valid))

# eval
print(model.evaluate(X_train, y_train)[1])
print(model.evaluate(X_valid, y_valid)[1])
print(model.evaluate(X_test, y_test)[1])

# acc graph
epoch = np.arange(1, 11)
acc_train = hist.history['accuracy']
acc_valid = hist.history['val_accuracy']
loss_train = hist.history['loss']
loss_valid = hist.history['val_loss']

plt.figure(figsize = (15,5))

plt.subplot(121)
plt.plot(epoch, acc_train, 'b', marker = '.', label = 'train_acc')
plt.plot(epoch, acc_valid, 'r--', marker = '.', label = 'valid_acc')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(122)
plt.plot(epoch, loss_train, 'b', marker = '.', label = 'train_loss')
plt.plot(epoch, loss_valid, 'r--', marker = '.', label = 'valid_loss')
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

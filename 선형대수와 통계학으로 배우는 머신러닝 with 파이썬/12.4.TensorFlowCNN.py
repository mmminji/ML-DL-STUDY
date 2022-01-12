import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

np.random.seed(0)
tf.random.set_seed(0)

# data load
(X_tn0, y_tn0), (X_te0, y_te0) = datasets.mnist.load_data()
print(X_tn0.shape)
print(y_tn0.shape)
print(X_te0.shape)
print(y_te0.shape)
print(set(y_tn0))

# data visualization
plt.figure(figsize = (10,5))
for i in range(2*5):
    plt.subplot(2,5,i+1)
    plt.imshow(X_tn0[i].reshape((28,28)), cmap='Greys')
plt.show()

# pre-processing
X_tn_re = X_tn0.reshape(60000, 28, 28, 1)   # (이미지 개수, 행, 열, 채널수) 합성곱 신경망은 4차원 형태로 입력
X_tn = X_tn_re/255
print(X_tn.shape)

X_te_re = X_te0.reshape(10000, 28, 28, 1)
X_te = X_te_re/255
print(X_te.shape)

y_tn = to_categorical(y_tn0)
y_te = to_categorical(y_te0)

# model
n_class = len(set(y_tn0))  # 최종 출력값

model = Sequential()
model.add(Conv2D(32, kernel_size = (5,5),        # 출력 차원, 커널 사이즈
                    input_shape = (28,28,1),     # 입력 차원
                    padding = 'valid',           # valid는 패딩 x, same은 입력 데이터와 동일한 크기로 조정
                    activation = 'relu'))        # 활성화 함수  
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size = (3,3), 
                    input_shape = (28,28,1), 
                    padding = 'valid', 
                    activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())                             # 벡터로 펼쳐줌
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation = 'softmax'))
model.summary()

# model compile
model.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

# train
hist = model.fit(X_tn, y_tn, epochs = 3, batch_size = 100)

# eval
print(model.evaluate(X_tn, y_tn)[1])
print(model.evaluate(X_te, y_te)[1])

# predict
y_pred_hot = model.predict(X_te)
y_pred = np.argmax(y_pred_hot, axis = 1)

# wrong predict visualization
diff = y_te0 - y_pred
diff_idx = []
y_len = len(y_te0)
for i in range(y_len):
    if(diff[i] != 0):
        diff_idx.append(i)

plt.figure(figsize = (10,5))
for i in range(2*5):
    plt.subplot(2,5,i+1)
    raw_idx = diff_idx[i]
    # print(y_pred[raw_idx])
    plt.imshow(X_te0[raw_idx].reshape((28,28)), cmap='Greys')
plt.show()
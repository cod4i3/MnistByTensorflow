from matplotlib import pyplot as plt
# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras
from keras import losses
from keras.utils.np_utils import to_categorical
import numpy as np
import time

t1 = time.time()

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) =\
    mnist.load_data()

inodes = 784
hnodes = 100
onodes = 10

# 正規化
train_images = train_images.reshape(60000, inodes).astype('float32')
test_images = test_images.reshape(10000, inodes).astype('float32')

# カテゴリ化
train_labels = to_categorical(train_labels, onodes)
test_labels = to_categorical(test_labels, onodes)


train_images, X_valid = np.split(train_images, [55000])
train_labels, Y_valid = np.split(train_labels, [55000])

train_images /= 255
test_images /= 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid')
])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


# ここのoptimizer, lossとかを弄れば大体OK
# mean_squared_error,categorical_crossentropy'
model.compile(optimizer=keras.optimizers.SGD(lr=0.3, momentum=0.0, decay=0.0, nesterov=False),
              loss='mean_squared_error',
              metrics=['acc'],
              callbacks=[callback]
              )

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(X_valid, Y_valid))
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)


t2 = time.time()
elapsed_time = t2 - t1
print(f"経過時間:{elapsed_time}")


# 精度のplot
plt.plot(history.history['acc'], marker='.', label="acc")
plt.plot(history.history['val_acc'], marker='.', label='val_acc')
plt.title('model accuracy')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

# 損失のplot
plt.plot(history.history['loss'], marker='.', label='loss')
plt.plot(history.history['val_loss'], marker='.', label='val_loss')
plt.title('model loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

import tensorflow as tf 
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt 


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
# plt.show()


# build the mode
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
''' Flatten layer transforms the images from 2d array (28, 28) to 1d array (28*28 = 784'''
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# make prediction
predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    pred = np.argmax(predictions[i])
    if(pred == test_labels[i]):
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(pred, test_labels[i]), color=color)
plt.show()
    
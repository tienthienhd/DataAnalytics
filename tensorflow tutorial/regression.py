import tensorflow as tf
from tensorflow import keras 

import numpy as np 
import matplotlib.pyplot as plt 

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# example and features
print('Training set: {}'.format(train_data.shape))
print('Testing set: {}'.format(test_data.shape))

# normalize features
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# create the model
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=64, activation=tf.nn.relu,
    input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))
    
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model
model = build_model()
model.summary()


history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, verbose=1)

def plot_history(history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Abs error[1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

plot_history(history)
[loss, mae] = model.evaluate(test_data, test_labels)
print("Testing set mean abs error: ${:7.2f}".format(mae*1000))

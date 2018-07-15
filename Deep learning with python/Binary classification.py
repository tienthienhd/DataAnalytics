
# coding: utf-8

# A binary classification example
# the IMDB dataset:
#     50 000 reviews from Internet Movie Database
#     25 000 for training : 50% negative and 50% positive
#     25 000 for testing  : 50% negative and 50% positive
#     

# In[1]:


from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words=10000 means keep the top 10 000 most frequently occurring words in the training data.


# In[2]:


word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decode_preview = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]])
decode_preview


# In[3]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[4]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[5]:


# config default with string parameter
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# # config parameters
# from keras import optimizers
# from keras import losses
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.01),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])


# In[6]:


# validating
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[7]:


model.reset_states()
history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=4,
                   batch_size=512,
                   validation_data=(x_val, y_val),
                   verbose=1)


# In[11]:


import matplotlib.pyplot as plt

history_dict = history.history
history_dict.keys()

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'b', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validataion loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() # clears the figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'b', label='Train acc')
plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[9]:


results = model.evaluate(x_test, y_test)
results


# In[10]:


model.predict(x_test)


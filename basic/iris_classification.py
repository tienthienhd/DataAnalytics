import pandas as pd 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

# preparing the data
dataset = pd.read_csv('F:\LabDataAnalytics\data\iris\iris.data', header=None)
dataset.columns= ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
print(dataset.head())

X_ = dataset.iloc[:, :4].values
# mean subtract and normalize
mean = X_.mean(axis=0)
std = X_.std(axis=0)
X_ = (X_ - mean) / std

y_ = dataset.iloc[:, 4].values

labels = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
y_ = [int(labels[x]) for x in y_]
y_ = np.reshape(y_, (len(y_,)))



# split to training and testing set
num_train = int(len(X_) * 0.8)
print('number of train:', num_train)
X_train = X_[:num_train]
y_train = y_[:num_train]

X_test = X_[num_train:]
y_test = y_[num_train:]

# build model
epochs = 20
batch_size = 10
learning_rate=0.01
total_batchs = int(num_train/batch_size)


x = tf.placeholder(dtype=tf.float32, name='x', shape=(None, 4))
y = tf.placeholder(dtype=tf.int32, name='y', shape=(None, ))

l1 = tf.layers.dense(inputs=x, units=4, activation=tf.nn.relu, name='l1')
l2 = tf.layers.dense(inputs=l1, units=4, activation=tf.nn.relu, name='l2')
pred = tf.layers.dense(inputs=l2, units=3, activation=tf.nn.softmax, name='output')

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=pred)

optitmizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir='./tmp/log/iris_graphs', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(total_batchs):
            x_batch = X_train[i*batch_size: (i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]

            predict, l, _ = sess.run([pred, loss, optitmizer], feed_dict={x:x_batch, y:y_batch})
            total_loss += l
        avg_loss = total_loss / total_batchs
        losses.append(avg_loss)
        print("Epoch #{} Avg_loss={}".format(epoch, avg_loss))

    predict, l = sess.run([pred, loss], feed_dict={x:X_test, y:y_test})
    print('Evaluate loss:', l)
    # plot loss
    epoch = range(1,epochs+1)
    plt.plot(epoch, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
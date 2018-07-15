
# coding: utf-8



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

learning_rate = 0.01
batch_size = 32

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis] # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise # shape (100, 1) + some noise

# plot dataset
plt.scatter(x, y)
plt.show()

# default network
class Net:
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        l = tf.layers.dense(self.x, 20, tf.nn.relu)
        out = tf.layers.dense(l, 1)
        self.loss = tf.losses.mean_squared_error(self.y, out)
        self.train = opt(learning_rate, **kwargs).minimize(self.loss)
        
# different nets
net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momemtum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)

nets = [net_SGD, net_Momemtum, net_RMSprop, net_Adam]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    losses_his = [[], [], [], []] # record loss
    
    # training
    for step in range(300):
        index = np.random.randint(0, x.shape[0], batch_size)
        b_x = x[index]
        b_y = y[index]
        
        for net, l_his in zip(nets, losses_his):
            _, l = sess.run([net.train, net.loss], {net.x: b_x, net.y:b_y})
            l_his.append(l) # loss recoder
            
    # plot loss history
    labels = ['SGD', 'Momemtum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()


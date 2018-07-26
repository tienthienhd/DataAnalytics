import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, sess, config):#, config):# full config
        self.sess = sess
        self.config(config)
        self.build_model()
        
        
    def config(self, config=None):
        sliding = config['sliding']
        
        self.learning_rate = config['learning_rate']
        
        
        self.input_dim = [sliding]
        self.output_dim = [1]
        
        self.epochs = config['epochs']
        
        self.activation = config['activation']
        if self.activation == 'relu':
            self.activation = tf.nn.relu
        elif self.activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif self.activation == 'tanh':
            self.activation = tf.nn.tanh
        
        self.batch_size = config['batch_size']
        
        self.optimizer = config['optimizer']
        if self.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer()
        elif self.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimizer == 'gd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate)
        
        
        
        self.hidden_layers = config['layers']
        
        
        
#        self.input_dim = [2]
#        self.output_dim = [1]
#        self.batch_size = 8
#        self.epochs =  100
#        self.hidden_layers = [2, 2]
#        self.activation = tf.nn.relu
#        self.optimizer = tf.train.AdamOptimizer
#        

        
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_dim, name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None] + self.output_dim, name='y')
        
        prev_layer = layer = None
        for i, num_units in enumerate(self.hidden_layers):
            if i==0:
                layer = tf.layers.dense(inputs=self.x, activation=self.activation, units=num_units, name='layer'+str(i))
            else:
                layer = tf.layers.dense(inputs=prev_layer, activation=self.activation, units=num_units, name='layer'+str(i))
            prev_layer = layer
        self.pred = tf.layers.dense(inputs=prev_layer, units=1, name='output_layer')
        
        with tf.variable_scope('loss'):        
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.pred)
    #        self.loss = tf.reduce_mean(tf.square(0.5*(self.pred - self.y) ** 2))
#            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.pred, self.y)))
            tf.summary.scalar("loss", self.loss)
            self.optimize = self.optimizer.minimize(self.loss)
        
        
    
    def predict(self, inputs):
        return self.sess.run([self.pred], feed_dict={self.x:inputs})
    
    
    def fit(self, x, y, x_val=None, y_val=None, verbose=1):
        
        loss_vals = []
        loss_trains = []
        
        for epoch in range(self.epochs):
            # training process
            loss_train = self.train(x, y)
            loss_trains.append(loss_train)
            
            
            # validating process
            if x_val is not None and y_val is not None:
               loss_val = self.evaluate(x_val, y_val)
               loss_vals.append(loss_val)
            
            if verbose == 1:
                if x_val is not None and y_val is not None:
                    print('Epoch #', epoch, 'training loss:', loss_train, ' ; validation loss:', loss_vals[-1])
                else:
                    print('Epoch #', epoch, 'training loss:', loss_train)
            
        return loss_trains, loss_vals
    
    
    def train(self, x, y):
        num_batchs = int(np.ceil(len(x)/self.batch_size))
        losses = []      
        for i in range(num_batchs):
            
            x_batch = x[i * self.batch_size : (i+1) * self.batch_size]
            y_batch = y[i * self.batch_size : (i+1) * self.batch_size]
            
            loss = self.train_step(x_batch, y_batch)
            losses.append(loss)
            
        return np.mean(losses)
    
    def train_step(self, x, y):
        feed_dict = {self.x:x, self.y:y}
        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=feed_dict)
        return loss
    
    
    def evaluate(self, x, y):
        num_batchs = int(np.ceil(len(x)/self.batch_size))
        losses = []      
        for i in range(num_batchs):
            
            x_batch = x[i * self.batch_size : (i+1) * self.batch_size]
            y_batch = y[i * self.batch_size : (i+1) * self.batch_size]
            
            loss = self.evaluate_step(x_batch, y_batch)
            losses.append(loss)
            
        return np.mean(losses)
    
    def evaluate_step(self, x, y):
        feed_dict = {self.x:x, self.y:y}
        loss = self.sess.run([self.loss], feed_dict=feed_dict)
        return loss
        
    
        
def test():
    x = list(range(1, 101, 1))
    x = np.reshape(x, [-1, 2])
    y = list(range(1, 51, 1))
    y = np.reshape(y, [-1, 1])
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    model = Model(sess)
    sess.run(tf.global_variables_initializer())
    loss_train, loss_val = model.fit(x, y)
    epochs = range(len(loss_train))
    
    plt.plot(epochs, loss_train, label='loss train')
    if len(loss_val) == len(epochs):
        plt.plot(epochs, loss_val, label='loss validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    
    sess.close()

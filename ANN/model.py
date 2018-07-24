import tensorflow as tf 
import numpy as np 

class Model(object):
    def __init__(self, sess, config):# full config
        self.sess = sess
        self.config(config)
        self.build_model()
        
        
    def config(self, config):
        sliding = config['sliding']
        
        
        self.input_shape_x = [sliding]
        self.input_shape_y = [1]
        
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
            self.optimizer = tf.train.AdamOptimizer
        elif self.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer
        elif self.optimizer == 'gd':
            self.optimizer = tf.train.GradientDescentOptimizer
        elif self.optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer
        elif self.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        
        
        
        self.hidden_layers = config['layers']
        
        self.learning_rate = config['learning_rate']
        
        

        
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_shape_x, name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_shape_y, name='y')
        
        prev_layer = layer = None
        for i, num_units in enumerate(self.hidden_layers):
            if i==0:
                layer = tf.layers.dense(inputs=self.x, activation=self.activation, units=num_units, name='layer'+str(i))
            else:
                layer = tf.layers.dense(inputs=prev_layer, activation=self.activation, units=num_units, name='layer'+str(i))
            prev_layer = layer
        self.pred = tf.layers.dense(inputs=prev_layer, units=1, name='output_layer')
        
        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.pred)
#        self.loss = tf.reduce_mean(tf.square(0.5*(self.pred - self.y) ** 2))
        tf.summary.scalar("loss", self.loss)
        self.optimize = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        
    
    def predict(self, inputs):
        return self.sess.run([self.pred], feed_dict={self.x:inputs})
    
    
    def train_model(self, x, y, x_val=None, y_val=None, verbose=1):
        if x_val is not None and y_val is not None:
            num_patchs_val = int(np.ceil(len(x_val)/self.batch_size))
            
            
        num_patchs_train = int(np.ceil(len(x)/self.batch_size))
        
        loss_vals = []
        loss_trains = []
        
        for epoch in range(self.epochs):
            # training process
            loss_train = []
            for i in range(num_patchs_train):
                x_ = x[i*self.batch_size:(i+1)*self.batch_size]
                y_ = y[i*self.batch_size:(i+1)*self.batch_size]
                
                loss_, optimizer = self.sess.run([self.loss, self.optimize], feed_dict={self.x:x_, self.y:y_})
                loss_train.append(loss_)
                
            loss_train = np.mean(loss_train)
            loss_trains.append(loss_train)
            
            
            # validating process
            if x_val is not None and y_val is not None:
                loss_val = []
                for j in range(num_patchs_val):
                    x_ = x_val[j*self.batch_size:(j+1)*self.batch_size]
                    y_ = y_val[j*self.batch_size:(j+1)*self.batch_size]
                    
                    loss_ = self.sess.run([self.loss], feed_dict={self.x:x_, self.y:y_})
                    loss_val.append(loss_)
                    
                loss_val = np.mean(loss_val)
                loss_vals.append(loss_val)
            
            if verbose == 1:
                if x_val is not None and y_val is not None:
                    print('Epoch #', epoch, 'training loss:', loss_train, ' ; validation loss:', loss_vals[-1])
                else:
                    print('Epoch #', epoch, 'training loss:', loss_train)
            
        epochs = list(range(self.epochs))
        return epochs, loss_trains, loss_vals
    




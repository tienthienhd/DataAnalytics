# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import Config
from data import Data



class MLP(object):
    def __init__(self, config):
        tf.reset_default_graph()
        self.hidden_layers = config['model_default_mlp']['hidden_layers']
        
        self.input_dim = 4 # equal num_units of encoder
        self.output_dim = [1] # predict from inputs to one output
        
        
        self.learning_rate = 0.001
            
        
        self.batch_size = config['model_default']['batch_size']
        self.num_epochs = config['model_default_mlp']['num_epochs']
        
        self.activation = config['model_default_mlp']['activation']
        if self.activation == 'relu':
            self.activation = tf.nn.relu
        elif self.activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif self.activation == 'tanh':
            self.activation = tf.nn.tanh
            
        self.optimizer = config['model_default_mlp']['optimizer']
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
        
        
        self.sess = tf.Session()
        self.load_encoder_model()
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        pass
    
    def build_model(self):
#        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_dim, name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None] + self.output_dim, name='y')
#        print(type(self.encoder_last_outputs))
#        print(self.encoder_last_outputs.shape)
        prev_layer = layer = None
        for i, num_units in enumerate(self.hidden_layers):
            if i==0:
                layer = tf.layers.dense(inputs=self.encoder_last_outputs, activation=self.activation, units=num_units, name='layer'+str(i))
            else:
                layer = tf.layers.dense(inputs=prev_layer, activation=self.activation, units=num_units, name='layer'+str(i))
            prev_layer = layer
        self.pred = tf.layers.dense(inputs=prev_layer, units=1, name='output_layer')
        
        with tf.variable_scope('loss'):        
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.pred)
    #        self.loss = tf.reduce_mean(tf.square(0.5*(self.pred - self.y) ** 2))
#            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.pred, self.y)))
#            tf.summary.scalar("loss", self.loss)
#            print(type(self.optimizer))
            self.optimize = self.optimizer.minimize(self.loss)
        pass
    
    def load_encoder_model(self):
        # load encoder_decoder model in default graph
        encoder_saver = tf.train.import_meta_graph('./log/model/encoder_decoder.ckpt.meta')
        
        encoder_graph = tf.get_default_graph()
        
        self.encoder_inputs = encoder_graph.get_tensor_by_name('encoder_x:0')
        encoder_outputs = encoder_graph.get_tensor_by_name('encoder/encoder_outputs:0')
        self.initial_state_encoder = encoder_graph.get_tensor_by_name('init_state:0')
        
        
        output_encoder_sg = tf.stop_gradient(encoder_outputs)
    
        self.encoder_last_outputs = output_encoder_sg[:, :, -1]
#        print(self.encoder_last_outputs)
        
        encoder_saver.restore(self.sess, './log/model/encoder_decoder.ckpt')
        with open('./log/model/state.pkl', 'rb') as f:
            self.init_state = pickle.load(f)
        
    
    def fit(self, data):
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            
            # training process
            total_loss = 0.0
            batch_index = 0
            while data.has_next('fit_ann', 'train'):
                batch_index += 1
                # generate batch
                x_batch, y_batch = data.next_batch('fit_ann', 'train')
                y_batch = np.reshape(y_batch, (len(y_batch), 1))
                
                # generate feed_dict
                feed_dict = {self.initial_state_encoder: self.init_state, 
                             self.encoder_inputs: x_batch,
                             self.y: y_batch}
                
                p, l, _ = self.sess.run([self.pred, self.loss, self.optimize], feed_dict=feed_dict)
                total_loss += l
               
            avg_loss_train = total_loss / batch_index    
            train_losses.append(avg_loss_train)
            data.reset_iterator('fit_ann')
            
            # validation precess
            total_loss = 0.0
            batch_index = 0
            while data.has_next('fit_ann', 'val'):
                batch_index += 1
                # generate batch
                x_batch, y_batch = data.next_batch('fit_ann', 'val')
                y_batch = np.reshape(y_batch, (len(y_batch), 1))
                
                # generate feed_dict
                feed_dict = {self.initial_state_encoder: self.init_state, 
                             self.encoder_inputs: x_batch,
                             self.y: y_batch}
                
                p, l = self.sess.run([self.pred, self.loss], feed_dict=feed_dict)
                total_loss += l
               
            avg_loss_val = total_loss / batch_index    
            val_losses.append(avg_loss_val)
            data.reset_iterator('fit_ann')
            
            if epoch % 10 == 0:
                print('Epoch #', epoch, 'train_loss=',avg_loss_train, 'val_loss=', avg_loss_val)
            
        plt.plot(range(self.num_epochs), train_losses, label='loss training')
        plt.plot(range(self.num_epochs), val_losses, label='loss validation')
        
        plt.legend()
        plt.show()
        
        
        data.reset_iterator('fit_ann')
        x_test, y_test = data.next_batch('fit_ann', 'test')
        pred = self.sess.run(self.pred, feed_dict={self.initial_state_encoder:self.init_state, self.encoder_inputs:x_batch})
        
#        print(y_test)
        num_examples = len(pred)
#        print(num_examples)
#        y_test = np.reshape(y_test, (num_examples))
        pred = np.reshape(pred, (num_examples))
        
#        print(y_batch)
#        print(pred)
        
        print(train_losses[-1])
        
        plt.plot(range(num_examples), y_test, label='atual')
        plt.plot(range(num_examples), pred, label='prediction')
        plt.xlabel('time')
        plt.ylabel('usage')
        plt.legend()
        plt.show()
        
        pass
    
    def predict(self):
        pass
    
    


config = Config('config.json')
model_config = config.get_model_config_mlp()
data_config = config.get_data_config()

data = Data(data_config)
data.series_to_supervised('fit_ann', model_config['model_default']['sliding_encoder'], 1)
data.init_iterator('fit_ann', batch_size=model_config['model_default']['batch_size'])

#data = Data(data_config)
#data.series_to_supervised('encoder', model_config['encoder_num_inputs'], model_config['encoder_num_outputs'])
#data.series_to_supervised('decoder', model_config['decoder_num_inputs'], model_config['decoder_num_outputs'])

#data.init_iterator('encoder', start_index=0, batch_size=16)
#data.init_iterator('decoder', start_index=0, batch_size=16)


model = MLP(model_config)
model.fit(data)
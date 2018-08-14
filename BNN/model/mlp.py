# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:39:46 2018

@author: HP Zbook 15
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_loss(train_losses, val_losses=None):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='train loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
#    plt.show()
    plt.savefig('./log/figure/test.png')

class MLP(object):
    def __init__(self, config=None, max_min=None):
        tf.reset_default_graph()
        self.hidden_layers = config['hidden_layers']
        self.activation = tf.nn.tanh
        self.num_epochs = config['num_epochs']
        
        self.batch_size = config['batch_size']
        self.optimizer = tf.train.AdamOptimizer()
        
        if max_min:
            self.max = max_min[0]
            self.min = max_min[1]
        
        
        self.sess = tf.Session()
        self.load_encoder(self.sess)
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        
        
        
    def load_encoder(self, sess):
        encoder_saver = tf.train.import_meta_graph('./log/model/encoder_decoder.ckpt.meta')
        
        encoder_graph = tf.get_default_graph()
        
        self.x = encoder_graph.get_tensor_by_name('encoder_x:0')
        
        encoder_outputs = encoder_graph.get_tensor_by_name('encoder/encoder_outputs:0')
        
        output_encoder_sg = tf.stop_gradient(encoder_outputs)
    
        self.encoder_last_outputs = output_encoder_sg[:, :, -1]
        
#        self.encoder_state = encoder_graph.get_tensor_by_name('init_state:0')
        
        
        encoder_saver.restore(sess, './log/model/encoder_decoder.ckpt')
#        with open('./log/model/state.pkl', 'rb') as f:
#            self.init_state = pickle.load(f)
        
    
    def early_stop(self, array, patience=0, min_delta=0.0):
        if len(array) <= patience :
            return False
        
        value = array[len(array) - patience - 1]
        arr = array[len(array)-patience:]
        check = 0
        for val in arr:
            if(val - value > min_delta):
                check += 1
        if(check == patience):
            return True
        else:
            return False
        
    def build_model(self):
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
        prev_layer = layer = None
        for i, num_units in enumerate(self.hidden_layers):
            if i==0:
                layer = tf.layers.dense(inputs=self.encoder_last_outputs, 
                                        activation=self.activation, 
                                        units=num_units, 
                                        name='layer'+str(i))
            else:
                layer = tf.layers.dense(inputs=prev_layer, 
                                        activation=self.activation, 
                                        units=num_units, 
                                        name='layer'+str(i))
            prev_layer = layer
        self.pred = tf.layers.dense(inputs=prev_layer, 
                                    units=1, 
                                    name='output_layer')
        
        pred_inverse = self.pred * (self.max + self.min) + self.min
        y_inverse = self.y * (self.max + self.min) + self.min
        
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(pred_inverse, 
                                                     y_inverse)))
        self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                tf.subtract(pred_inverse, y_inverse))))
        
        with tf.device('/device:GPU:0'):
            with tf.variable_scope('loss'):        
    #            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.pred)
        #        self.loss = tf.reduce_mean(tf.square(0.5*(self.pred - self.y) ** 2))
                self.loss = tf.reduce_mean(tf.square(tf.subtract(self.pred, self.y)))
    #            tf.summary.scalar("loss", self.loss)
    #            print(type(self.optimizer))
                self.optimize = self.optimizer.minimize(self.loss)

    def fit(self, train, val=None, test=None):
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            
            avg_loss_train = self.train(train)
            train_losses.append(avg_loss_train)
            
            avg_loss_val = self.validate(val)
            val_losses.append(avg_loss_val)
            
            print('Epoch #%d train loss=%.7f val loss=%.7f' % (epoch, 
                  avg_loss_train, avg_loss_val))
            
            if self.early_stop(val_losses, 5):
                    print('finished training at epoch', epoch)
                    break
            
        if val:
            plot_loss(train_losses, val_losses)          
        else:
            plot_loss(train_losses)
            
        if test:                    
            self.test(test)
        
    
    def train(self, data):
        x = data[0]
        y = data[1]
        
        num_batches = 0
        total_loss = 0.0
        
        try:
            while True:
                x_ = x[num_batches * self.batch_size : 
                    (num_batches + 1) * self.batch_size]
                    
                y_ = y[num_batches * self.batch_size : 
                    (num_batches + 1) * self.batch_size]
                    
                _loss, o = self.sess.run([self.loss, self.optimize], feed_dict={
                        self.x: x_, 
                        self.y: y_
#                        self.encoder_state: self.init_state
                        })
    
                total_loss += _loss
                num_batches += 1
        except tf.errors.InvalidArgumentError:
            pass
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, data):
        x = data[0]
        y = data[1]
        
        num_batches = 0
        total_loss = 0.0
        
        try:
            while True:
                x_ = x[num_batches * self.batch_size : 
                    (num_batches + 1) * self.batch_size]
                    
                y_ = y[num_batches * self.batch_size : 
                    (num_batches + 1) * self.batch_size]
                    
                _loss = self.sess.run(self.loss, feed_dict={
                        self.x: x_, 
                        self.y: y_
#                        self.encoder_state: self.init_state
                        })
    
                total_loss += _loss
                num_batches += 1
        except tf.errors.InvalidArgumentError:
            pass
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def test(self, data):
        x = data[0]
        y = data[1]
        
        mae = []
        rmse = []
        num_batches = 0
        total_loss = 0.0
        
        try:
            while True:
                x_ = x[num_batches * self.batch_size : 
                    (num_batches + 1) * self.batch_size]
                    
                y_ = y[num_batches * self.batch_size : 
                    (num_batches + 1) * self.batch_size]
                    
                _mae, _rmse, _loss = self.sess.run([self.MAE, self.RMSE, self.loss], 
                                       feed_dict={
                                            self.x: x_, 
                                            self.y: y_
#                                            self.encoder_state: self.init_state
                                            })
                
                mae.append(_mae)
                rmse.append(_rmse)
                total_loss += _loss
                num_batches += 1
        except tf.errors.InvalidArgumentError:
            pass
        mae = np.mean(mae)
        rmse = np.mean(rmse)
        avg_loss = total_loss / num_batches
        print('loss: %.7f  mae: %.7f  rmse: %.7f' % (avg_loss, mae, rmse))
         
    
#mlp = MLP()
#sess = tf.Session()
##mlp.load_encoder(sess)
#mlp.fit()
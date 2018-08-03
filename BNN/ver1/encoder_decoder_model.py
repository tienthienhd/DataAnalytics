# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def multi_lstm(num_layers, num_units, keep_prob=1):
    cells = []
    for i in range(num_layers):
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)

def plot_loss(train_losses, val_losses=None):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='train loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

class EncoderDecoder(object):
    def __init__(self, config, data):
        self.encoder_num_inputs = config['encoder_num_inputs']
        self.decoder_num_inputs = config['decoder_num_inputs']
        self.encoder_num_outputs = config['encoder_num_outputs']
        self.decoder_num_outputs = config['decoder_num_outputs']
        
        self.batch_size = config['batch_size']
        self.keep_prob = config['keep_prob']
        
        self.num_units = config['num_units']
        self.num_layers = config['num_layers']
        self.num_epochs = config['num_epochs']
        self.num_features = data.get_num_features()
        
        tf.reset_default_graph()
        
    
        
    def build_model(self):
        
        # input for model
        self.encoder_x = tf.placeholder(dtype=tf.float32, shape=[None, self.encoder_num_inputs, self.num_features], name='encoder_x') # (batch_size, sequence_length, num_feature)
        
        self.decoder_x = tf.placeholder(dtype=tf.float32, shape=[None, self.decoder_num_inputs, self.num_features], name='decoder_x')
        
        self.decoder_y = tf.placeholder(dtype=tf.float32, shape=[None, self.decoder_num_outputs], name='decoder_y')
        
        self.init_state = tf.placeholder(dtype=tf.float32, shape=[self.num_layers, 2, self.batch_size, self.num_units], name='init_state')
        # 2 is c and h
        
        
        
        with tf.variable_scope('encoder'):
            
            # get tuple state for initialize encoder
            state_per_layer_list = tf.unstack(self.init_state, axis=0)
            rnn_tuple_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], 
                                                   state_per_layer_list[idx][1])
                    for idx in range(self.num_layers)]
                    )
                    
    #        print(rnn_tuple_state)
            
            encoder_cell = multi_lstm(self.num_layers, self.num_units, keep_prob=self.keep_prob)
            
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, 
                                                               inputs=self.encoder_x,
                                                               initial_state=rnn_tuple_state,
                                                               dtype=tf.float32)
            
        with tf.variable_scope('decoder'):
            decoder_cell = multi_lstm(self.num_layers, self.num_units, keep_prob=self.keep_prob)
            
            self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, 
                                                               inputs=self.decoder_x,
                                                               initial_state=self.encoder_state)
            
#            pred = decoder_outputs[:,  -1, :]
    #        print(pred.shape)
            self.pred_decoder = tf.layers.dense(inputs=self.decoder_outputs[:, -1, :], units=self.decoder_num_outputs, name='dense_output')
        
    def train_model(self, data):

        self.build_model()
        
        with tf.name_scope('loss_optimizer'):
            loss = tf.reduce_mean(tf.squared_difference(self.pred_decoder, self.decoder_y))
            optimizer  = tf.train.AdamOptimizer().minimize(loss)
            
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(init)
            self._current_state = tf.zeros((self.num_layers, 2, self.batch_size, self.num_units))
#            writer = tf.summary.FileWriter('./log', sess.graph)
            
            loss_train = []
            loss_val = []
            
            data.init_iterator('encoder', start_index=0, batch_size=self.batch_size)
            data.init_iterator('decoder', start_index=4, batch_size=self.batch_size)
            
            
            for epoch in range(self.num_epochs):
                epoch_loss_train = self.train(sess, data, optimizer, loss)
                loss_train.append(epoch_loss_train)
                
                epoch_loss_val = self.evaluate(sess, data, loss)
                loss_val.append(epoch_loss_val)
                print('Epoch #{}: loss_train={}  loss_val={}'.format(epoch, epoch_loss_train, epoch_loss_val))
            
            saver.save(sess, './log/model/encoder_decoder.ckpt')
            plot_loss(loss_train, loss_val)
    
    def train(self, sess, data, optimizer, loss):
        total_loss = 0.0
        batch_index = 0
        while data.has_next('encoder', 'train') and data.has_next('decoder', 'train'):
            batch_index += 1
            # generate batch
            encoder_x_batch, encoder_y_batch = data.next_batch('encoder', 'train')
            decoder_x_batch, decoder_y_batch = data.next_batch('decoder', 'train')
            
            # generate feed_dict
            feed_dict = {self.encoder_x:encoder_x_batch,
                         self.decoder_x:decoder_x_batch,
                         self.decoder_y:decoder_y_batch,
                         self.init_state:self._current_state}
        
            # fit batch and get loss batch
            _, loss_batch, self._current_state = sess.run([optimizer, loss, self.encoder_state], feed_dict=feed_dict)
            
            total_loss += loss_batch
        # calculate total loss
        avg_loss = total_loss / batch_index
        
        data.reset_iterator('encoder')
        data.reset_iterator('decoder')
        return avg_loss
#    
    def evaluate(self, sess, data, loss):
        total_loss = 0.0
        batch_index = 0
        while data.has_next('encoder', 'val') and data.has_next('decoder', 'val'):
            batch_index += 1
            # generate batch
            encoder_x_batch, encoder_y_batch = data.next_batch('encoder', 'val')
            decoder_x_batch, decoder_y_batch = data.next_batch('decoder', 'val')
            
            # generate feed_dict
            feed_dict = {self.encoder_x:encoder_x_batch,
                         self.decoder_x:decoder_x_batch,
                         self.decoder_y:decoder_y_batch,
                         self.init_state:self._current_state}
        
            # fit batch and get loss batch
            loss_batch, _current_state = sess.run([loss, self.encoder_state], feed_dict=feed_dict)
            
            total_loss += loss_batch
        # calculate total loss
        avg_loss = total_loss / batch_index
        return avg_loss
    
    
#    def fit_encoder(self, inputs):
#        self.build_model()
#        
#        saver = tf.train.Saver()
#        
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            
#            print('Loading variables...')
#            saver.restore(sess, './log/model/encoder_decoder.ckpt')
#            
#            feed_dict = {
#                    self.encoder_x:inputs,
##                    self.decoder_x:None,
##                    self.decoder_y:None,
#                    self.init_state:}
#            
#            outputs = sess.run([self.encoder_outputs], 
#                               feed_dict=feed_dict)
#            return outputs
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
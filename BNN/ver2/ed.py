# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:26:19 2018

@author: HP Zbook 15
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data import Data

tf.reset_default_graph()


class EncoderDecoder(object):
    def __init__(self, config=None):
        self.encoder_sliding = 4
        self.decoder_sliding = 4
        self.num_layers = 2
        self.num_units = 4
        self.keep_prob = 0.8
        self.batch_size = 8
        self.num_features = 1
        self.num_epochs = 100
    
    def _multi_lstm(self, num_layers, num_units, keep_prob=1):
        cells = []
        for i in range(num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells.append(cell)
        return tf.nn.rnn_cell.MultiRNNCell(cells)
    
    def make_iterator_data(self):
        # placeholder for input data    
        self.data_encoder_x = tf.placeholder(dtype=tf.float32, shape=[None, self.encoder_sliding, self.num_features], name='data_encoder_x')
        self.data_decoder_x = tf.placeholder(dtype=tf.float32, shape=[None, self.decoder_sliding, self.num_features], name='data_decoder_x')
        self.data_decoder_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='data_decoder_y')
        
        # create dataset of tensorflow
        dataset_encoder = tf.data.Dataset.from_tensor_slices(self.data_encoder_x)
        dataset_decoder = tf.data.Dataset.from_tensor_slices((self.data_decoder_x, self.data_decoder_y))
        
        # initial batch_size
        dataset_encoder = dataset_encoder.batch(self.batch_size)
        dataset_decoder = dataset_decoder.batch(self.batch_size)
        
#        # initial repeat dataset for epochs
#        dataset_encoder = dataset_encoder.repeat(self.num_epochs)
#        dataset_decoder = dataset_decoder.repeat(self.num_epochs)
#        
        # make iterator
        iterator_encoder = dataset_encoder.make_initializable_iterator()
        iterator_decoder = dataset_decoder.make_initializable_iterator()
        
        return iterator_encoder, iterator_decoder
    
    
    def build_model(self):
        iterator_encoder, iterator_decoder = self.make_iterator_data()
        
        # input for model
        encoder_x = iterator_encoder.get_next()
        decoder_x, decoder_y = iterator_decoder.get_next()
        
        self.init_state = tf.placeholder(dtype=tf.float32, shape=[self.num_layers, 2, self.batch_size, self.num_units], name='init_state')
        
        # init weight / bias output
        weight_out = tf.Variable(tf.random_normal([self.num_units, 1]), name='weight_out')
        bias_out = tf.Variable(tf.random_normal([1]), name='bias')
        
        # build model encoder
        with tf.variable_scope('encoder'):
            # get tuple state for initialize encoder
            state_per_layer_list = tf.unstack(self.init_state, axis=0)
            rnn_tuple_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], 
                                                   state_per_layer_list[idx][1])
                    for idx in range(self.num_layers)]
                    )
                    
        #        print(rnn_tuple_state)
            
            encoder_cell = self._multi_lstm(self.num_layers, self.num_units, self.keep_prob)
            
            encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, 
                                                               inputs=encoder_x,
                                                               initial_state=rnn_tuple_state,
                                                               dtype=tf.float32)
        
        # build model decoder
        with tf.variable_scope('decoder'):
            decoder_cell = self._multi_lstm(self.num_layers,self.num_units, self.keep_prob)
            
            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, 
                                                               inputs=decoder_x,
                                                               initial_state=self.encoder_state)
            
            decoder_outputs_last = decoder_outputs[:, -1, :]
            pred_decoder = tf.matmul(decoder_outputs_last, weight_out) + bias_out
        
        return iterator_encoder, iterator_decoder, pred_decoder, decoder_y
                
    
    
ed = EncoderDecoder()

#config = Config('config.json').get_config()            
#data = Data(config)
#data.series_to_supervised('encoder', config['encoder_decoder']['encoder_sliding'], 0)
#data.series_to_supervised('decoder', config['encoder_decoder']['decoder_sliding'], 1)
#
#e_x, e_y = data.get_data('encoder', 'train')
#d_x, d_y = data.get_data('decoder', 'train')
#
#train = (e_x, d_x, d_y)
#
#ed.train_model(train)
 
ed.build_model()           
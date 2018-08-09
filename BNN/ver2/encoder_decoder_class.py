# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import Config
from data import Data

def plot_loss(train_losses, val_losses=None):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='train loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('./log/figure/test.svg')
#    plt.show()
    


tf.reset_default_graph()

class EncoderDecoder(object):
    def __init__(self, config=None, max_min=None):
        self.encoder_sliding = config['encoder_decoder']['encoder_sliding']
        self.decoder_sliding = config['encoder_decoder']['decoder_sliding']
        self.num_layers = config['encoder_decoder']['num_layers']
        self.num_units = config['encoder_decoder']['num_units']
        self.keep_prob = config['encoder_decoder']['keep_prob']
        self.batch_size = config['encoder_decoder']['batch_size']
        self.num_features = config['encoder_decoder']['num_features']
        self.num_epochs = config['encoder_decoder']['num_epochs']
        
        if max_min:
            self.max_y = max_min[0]
            self.min_y = max_min[1]
        
    def _multi_lstm(self, num_layers, num_units, keep_prob=1):
        cells = []
        for i in range(num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells.append(cell)
        return tf.nn.rnn_cell.MultiRNNCell(cells)
       
    
    def build_model(self, data):
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        # input for model
        encoder_x, decoder_x, decoder_y = iterator.get_next()
#        decoder_x, decoder_y = iterator_decoder.get_next()
        
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
            
            pred_inverse = pred_decoder * (self.max_y + self.min_y) + self.min_y
            y_inverse = decoder_y * (self.max_y + self.min_y) + self.min_y
            
            self.MAE = tf.reduce_mean(tf.abs(tf.subtract(pred_inverse, y_inverse)))
            self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred_inverse, y_inverse))))
            
            
        return pred_decoder, decoder_y, iterator #iterator_encoder, iterator_decoder, 
    
    
    def train_model(self, train_set, val_set=None, test_set=None):
        predict, actual, iterator = self.build_model(train_set)
        
        with tf.device('/device:GPU:0'):
            with tf.name_scope('loss_optimizer'):
                loss = tf.reduce_mean(tf.square(predict - actual))
                optimizer  = tf.train.AdamOptimizer().minimize(loss)
        
        
        train_init_op = iterator.make_initializer(train_set)
        val_init_op = iterator.make_initializer(val_set)
        test_init_op = iterator.make_initializer(test_set)
        
        
        initialize = tf.global_variables_initializer()
        
#        saver = tf.train.Saver()
        
#        
#        batches_per_epoch = int(train[2].shape[0] / self.batch_size)+1
#        print(batches_per_epoch)
        
        _current_state = np.zeros((self.num_layers, 2, self.batch_size, self.num_units))
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(initialize)
            
            
            train_losses= []
            val_losses = []
            
            for epoch in range(self.num_epochs):
                
                
                # training 
                sess.run(train_init_op)
                avg_loss_train = 0.0
                num_batchs = 0
                try:
                    while True:
                        _current_state, l, o = sess.run([self.encoder_state, loss, optimizer], feed_dict={self.init_state: _current_state})
                        avg_loss_train += l 
                        num_batchs += 1
                except tf.errors.InvalidArgumentError:
                    pass  
                avg_loss_train /= num_batchs
                train_losses.append(avg_loss_train)
                
                
                ################################################
                
                if val_set:
                # validating
                    sess.run(val_init_op)
    
                    avg_loss_val = 0.0
                    num_batchs = 0
                    try:
                        while True:
                            l = sess.run(loss, feed_dict={self.init_state: _current_state})
                            avg_loss_val += l
                            num_batchs += 1
                    except tf.errors.InvalidArgumentError:
                        pass
                    avg_loss_val /= num_batchs
                    val_losses.append(avg_loss_val)
                print('Epoch #', epoch, 'loss = ', avg_loss_train, 'val_loss=', avg_loss_val)
                      
                    
                
            plot_loss(train_losses, val_losses)
            
            sess.run(test_init_op) 
            mae = []
            rmse = []
            try:
                while True:
                        
                    _mae, _rmse = sess.run([self.MAE, self.RMSE], feed_dict={self.init_state: _current_state})
                    mae.append(_mae)
                    rmse.append(_rmse)
            except tf.errors.InvalidArgumentError:
                pass
            
            mae = np.sum(mae) / len(mae)
            rmse = np.sum(rmse) / len(rmse)
            print(mae, rmse)
            with open('./log/mae_rmse.txt', 'w+') as f:
                f.write('{},{}\n'.format(mae, rmse))
            
            
            
start_time = datetime.now()
config = Config('config.json').get_config()            
data = Data(config)
data.prepare_data_inputs(config['encoder_decoder']['encoder_sliding'], config['encoder_decoder']['decoder_sliding'])

train_set = tf.data.Dataset.from_tensor_slices(data.get_data('train')).batch(config['encoder_decoder']['batch_size'])
val_set = tf.data.Dataset.from_tensor_slices(data.get_data('val')).batch(config['encoder_decoder']['batch_size'])
test_set = tf.data.Dataset.from_tensor_slices(data.get_data('test')).batch(config['encoder_decoder']['batch_size'])


ed = EncoderDecoder(config, data.get_max_min('meanCPUUsage'))


ed.train_model(train_set, val_set, test_set)

#pred = data.denormalize(pred, 'meanCPUUsage')
#actual = data.denormalize(actual, 'meanCPUUsage')
#
#loss = np.mean(np.sqrt(np.square(np.subtract(pred, actual))))
#print(loss/data.max['meanCPUUsage'])



end_time = datetime.now()

print(end_time-start_time)
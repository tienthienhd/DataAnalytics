# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 08:56:40 2018

@author: HP Zbook 15
"""
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from datetime import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd



def multi_lstm(layers_units, keep_prob=1):
    cells = []
    for num_units in layers_units:
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)

def plot_loss(train_losses, val_losses=None, file_save=None):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='train loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(file_save)
#    plt.show()
    plt.clf()



class EncoderDecoder(object):
    
    def __init__(self, config, max_min_data=None):
        tf.reset_default_graph()
        self.encoder_sliding = config['sliding'][0]
        self.decoder_sliding = config['sliding'][1]
        
#        self.num_layers = config.encoder_decoder['num_layers']
#        self.num_units = config.encoder_decoder['num_units']
        self.layers_units = config['layers_units']
        
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.keep_prob = config['keep_prob']
        self.num_features = config['num_features']
        
        if max_min_data:
            self.max = max_min_data[0]
            self.min = max_min_data[1]
        
        
    
        
    def build_model(self):
        self.encoder_x = tf.placeholder(dtype=tf.float32, 
                           shape=[None, self.encoder_sliding, self.num_features],
                           name='encoder_x')
        self.decoder_x = tf.placeholder(dtype=tf.float32, 
                           shape=[None, self.decoder_sliding, self.num_features],
                           name='decoder_x')
        self.decoder_y = tf.placeholder(dtype=tf.float32,
                                   shape=[None, 1], 
                                   name='decoder_y')
        
#        self.init_state = tf.placeholder(dtype=tf.float32,
#                                    shape=[self.num_layers, 2, self.batch_size,
#                                           self.num_units], name='init_state')
    
        with tf.variable_scope('encoder'):
#            # get tuple state for initialize encoder
#            state_per_layer_list = tf.unstack(self.init_state, axis=0)
#            rnn_tuple_state = tuple(
#                    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], 
#                                                   state_per_layer_list[idx][1])
#                    for idx in range(self.num_layers)]
#                    )
            
            encoder_cell = multi_lstm(self.layers_units, 
                                      keep_prob=self.keep_prob)
            
            with tf.variable_scope('hidden_state'):
                state_variables = []
                for state_c, state_h in encoder_cell.zero_state(self.batch_size, tf.float32):
                    state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                        tf.Variable(state_c, trainable=False),
                        tf.Variable(state_h, trainable=False)))
                # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
                rnn_tuple_state = tuple(state_variables)
            
            
            
            
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, 
                                               inputs=self.encoder_x,
                                               initial_state=rnn_tuple_state,
                                               dtype=tf.float32)
            

            # Define an op to keep the hidden state between batches
            update_ops = []
            for state_variable, new_state in zip(rnn_tuple_state, encoder_state):
                # Assign the new state to the state variables on this layer
                update_ops.extend([state_variable[0].assign(new_state[0]),
                                   state_variable[1].assign(new_state[1])])
            # Return a tuple in order to combine all update_ops into a single operation.
            # The tuple's actual value should not be used.
            self.rnn_keep_state_op = tf.tuple(update_ops)     
            
            
            
            encoder_outputs = tf.identity(encoder_outputs, 
                                          name='encoder_outputs')
            
        with tf.variable_scope('decoder'):
            decoder_cell = multi_lstm(self.layers_units,
                                      keep_prob=self.keep_prob)
            
            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, 
                                                   inputs=self.decoder_x,
                                                   initial_state=encoder_state)
            pred_decoder = tf.layers.dense(inputs=decoder_outputs[:, -1, :], 
                                           units=1, 
                                           name='dense_output')      
            pred_decoder = tf.identity(pred_decoder, 'decoder_pred')
            
            self.pred_inverse = pred_decoder * (self.max + self.min) + self.min
            self.y_inverse = self.decoder_y * (self.max + self.min) + self.min
            
            self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.pred_inverse, 
                                                         self.y_inverse)))
            self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                    tf.subtract(self.pred_inverse, self.y_inverse))))
    
#        with tf.device('/device:GPU:0'):
        with tf.name_scope('loss_optimizer'):
            loss = tf.reduce_mean(tf.squared_difference(pred_decoder, 
                                                        self.decoder_y))
            optimizer  = tf.train.AdamOptimizer().minimize(loss)
            
            
        return encoder_state, pred_decoder, loss, optimizer
                
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
        
    
    def fit(self, train, val=None, test=None, folder_result=None, config_name=None):
        
        if folder_result and config_name:
            history_file = folder_result + config_name + '_history.png'
            error_file = folder_result + config_name + '_error.csv'
            predict_file = folder_result + config_name + '_predict.csv'
            model_file = folder_result + config_name + '_model_encoder_decoder.ckpt'
            mae_rmse_file = folder_result + 'mae_rmse_log.csv'
        
        encoder_state, pred_decoder, loss, optimizer = self.build_model()
        
#        _current_state = np.zeros((self.num_layers, 
#                                   2, 
#                                   self.batch_size, 
#                                   self.num_units))
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
#            writer = tf.summary.FileWriter('./log/graph/', sess.graph)
            
            train_losses = []
            val_losses = []
            
            for epoch in range(self.num_epochs):
                start_time = datetime.now()
#                print('start epoch #', epoch)
                _current_state, avg_loss_train = self._train(sess, train, 
#                                                             _current_state, 
                                                             encoder_state, 
                                                             loss, optimizer)
                train_losses.append(avg_loss_train)
#                break
                
                if val:
                    avg_loss_val = self._validate(sess, val, loss)
                    val_losses.append(avg_loss_val)                    
#                    print('Epoch #%d loss train = %.7f  loss_val = %.7f' % (epoch,
#                          avg_loss_train, avg_loss_val))
                else:
#                    print('Epoch #%d loss train = %.7f' % (epoch,
#                         avg_loss_train))
#                print('interval time:', 
#                      datetime.now()-start_time)
				    pass
                
                if self.early_stop(val_losses, 5):
                    print('finished training at epoch', epoch)
                    break
                
            
            
            if val:
                # log result
                if folder_result and config_name:
                    log = {'train': train_losses, 'val': val_losses}
                    df_log = pd.DataFrame(log)
                    df_log.to_csv(error_file, index=None)
                    
                plot_loss(train_losses, val_losses, history_file)          
            else:
                plot_loss(train_losses)
            if test:                    
                self._test(sess, test, loss, predict_file, mae_rmse_file)
                
            saver.save(sess, model_file)
#            with open('./log/model/state.pkl', 'wb') as f:
#                pickle.dump(_current_state, f)
            
            
                
                
            
                
                
                
    def _train(self, sess, data, encoder_state, loss, optimizer):
        encoder_x = data[0]
        decoder_x = data[1]
        decoder_y = data[2]
        
        num_batches = 0
        total_loss = 0.0
        try:
            while True:
                e_x = encoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_x = decoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_y = decoder_y[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                
                
                
                _current_state, _loss, o, _ = sess.run([encoder_state, loss, optimizer, self.rnn_keep_state_op],
                                        feed_dict={self.encoder_x: e_x,
                                                   self.decoder_x: d_x,
                                                   self.decoder_y: d_y
#                                                   self.init_state: _current_state
                                                })
    
#                sess.run(self.rnn_keep_state_op)
#                print(_current_state)
                total_loss += _loss
                num_batches += 1
        except tf.errors.InvalidArgumentError:
            pass
        avg_loss = total_loss / num_batches
        return _current_state, avg_loss
            
            
        
    def _validate(self, sess, data, loss):
        encoder_x = data[0]
        decoder_x = data[1]
        decoder_y = data[2]
        
        num_batches = 0
        total_loss = 0.0
        try:
            while True:
                e_x = encoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_x = decoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_y = decoder_y[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                
                
                
                _loss = sess.run(loss,feed_dict={self.encoder_x: e_x,
                                                   self.decoder_x: d_x,
                                                   self.decoder_y: d_y
#                                                   self.init_state: _current_state
                                                })
                total_loss += _loss
                num_batches += 1
        except tf.errors.InvalidArgumentError:
            pass
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _test(self, sess, data, loss, log_file=None, log_mae_rmse=None):
        encoder_x = data[0]
        decoder_x = data[1]
        decoder_y = data[2]
        
        mae = []
        rmse = []
        total_loss = 0.0
        num_batches = 0
        
        predict = []
        actual = []
        try:
            while True:
                e_x = encoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_x = decoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_y = decoder_y[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                    
                pred_inv, y_inv, _mae, _rmse, _loss = sess.run([self.pred_inverse, self.y_inverse, self.MAE, self.RMSE, loss], 
                                             feed_dict={
                                                self.encoder_x: e_x,
                                                self.decoder_x: d_x,
                                                self.decoder_y: d_y
#                                                self.init_state: _current_state
                                            })
                mae.append(_mae)
                rmse.append(_rmse)
                total_loss += _loss
                num_batches += 1
                
                predict.extend(pred_inv[:, 0])
                actual.extend(y_inv[:, 0])
        except tf.errors.InvalidArgumentError:
            pass
        mae = np.mean(mae)
        rmse = np.mean(rmse)
        avg_loss = total_loss / num_batches
        print('loss: %.7f  mae: %.7f  rmse: %.7f' % (avg_loss, mae, rmse))
        
        with open(log_mae_rmse, 'a+') as f:
            f.write('%f, %f\n' % (mae, rmse))
            
        
        log = {'predict': predict, 'actual': actual}
        df_log = pd.DataFrame(log)
        df_log.to_csv(log_file, index=None)
#    def predict(self)                
                
            
        
        
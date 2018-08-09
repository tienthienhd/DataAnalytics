import tensorflow as tf 
import numpy as np 
import pandas as pd
import math
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from data import Data
from config import Config
from tensorflow.python.client import timeline


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
    plt.savefig('./log/figure/test.png')


class EncoderDecoder(object):
    def __init__(self, config):
        self.encoder_sliding = config['model_default']['sliding_encoder']
        self.decoder_sliding = config['model_default_encoder']['sliding_decoder']
        
        self.batch_size = config['model_default']['batch_size']
        self.keep_prob = config['model_default_encoder']['keep_prob']
        
        self.num_layers = config['model_default_encoder']['num_layers']
        self.num_units = config['model_default_encoder']['num_units']
        self.num_epochs = config['model_default_encoder']['num_epochs']
        self.num_features = len(config['model_default']['features'])
        
        tf.reset_default_graph()
        
        
    def build_model(self):
        # input for model
        self.encoder_x = tf.placeholder(dtype=tf.float32, shape=[None, self.encoder_sliding, self.num_features], name='encoder_x') # (batch_size, sequence_length, num_feature)
        
        self.decoder_x = tf.placeholder(dtype=tf.float32, shape=[None, self.decoder_sliding, self.num_features], name='decoder_x')
        
        self.decoder_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='decoder_y')
        
        self.init_state = tf.placeholder(dtype=tf.float32, shape=[self.num_layers, 2, self.batch_size, self.num_units], name='init_state')
        # 2 is c and h
        
#        self.current_encoder_state = tf.get_variable("current_encoder_state", dtype=tf.float32, 
#                                                shape=[self.num_layers, 2, self.batch_size, self.num_units])
        
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
            
            self.encoder_outputs = tf.identity(self.encoder_outputs, 'encoder_outputs')
            
        with tf.variable_scope('decoder'):
            decoder_cell = multi_lstm(self.num_layers, self.num_units, keep_prob=self.keep_prob)
            
            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, 
                                                               inputs=self.decoder_x,
                                                               initial_state=self.encoder_state)
            
            self.pred_decoder = tf.layers.dense(inputs=decoder_outputs[:, :, -1], units=1, name='dense_output')      
            self.pred_decoder = tf.identity(self.pred_decoder, 'decoder_pred')
            
    
    def train_model(self, data):
        self.build_model()
        
        with tf.name_scope('loss_optimizer'):
            loss = tf.reduce_mean(tf.squared_difference(self.pred_decoder, self.decoder_y))
            optimizer  = tf.train.AdamOptimizer().minimize(loss)
        
        saver = tf.train.Saver()
        
        
        self._current_state = np.zeros((self.num_layers, 2, self.batch_size, self.num_units))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
#            writer = tf.summary.FileWriter('./log/graph/', sess.graph)
            
            train_losses= []
            val_losses= []
            
            data.init_iterator('encoder', start_index=0, batch_size=self.batch_size)
            data.init_iterator('decoder', start_index=4, batch_size=self.batch_size)
            
            for epoch in range(self.num_epochs):
                
                # training process
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
                avg_loss_train = total_loss / batch_index
                data.reset_iterator('encoder')
                data.reset_iterator('decoder')
                
                train_losses.append(avg_loss_train)
                
                
                # validation process
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
                    _, loss_batch = sess.run([optimizer, loss], feed_dict=feed_dict)
                    total_loss += loss_batch
                # calculate total loss
                avg_loss_val = total_loss / batch_index
                data.reset_iterator('encoder')
                data.reset_iterator('decoder')
                
                val_losses.append(avg_loss_val)
                
                if epoch % 10 == 0:
                    print('Epoch #{}: loss_train={} loss_val={}'.format(epoch, avg_loss_train, avg_loss_val))
            
            # saving model
            saver.save(sess, './log/model/encoder_decoder.ckpt')
            with open('./log/model/state.pkl', 'wb') as f:
                pickle.dump(self._current_state, f)
            
            # loss test
            
            
            plot_loss(train_losses, val_losses) 
            
            
            
            
            
            
   
            
            
                                
    def evaluate(self, data):
        
        data.init_iterator('encoder', start_index=0, batch_size=self.batch_size)
        data.init_iterator('decoder', start_index=4, batch_size=self.batch_size)
         # load encoder_decoder model in default graph
        encoder_saver = tf.train.import_meta_graph('./log/model/encoder_decoder.ckpt.meta')
        
        encoder_graph = tf.get_default_graph()
        
        encoder_inputs = encoder_graph.get_tensor_by_name('encoder_x:0')
        decoder_inputs = encoder_graph.get_tensor_by_name('decoder_x:0')
        decoder_outputs = encoder_graph.get_tensor_by_name('decoder_y:0')
        initial_state_encoder = encoder_graph.get_tensor_by_name('init_state:0')
        pred_decoder = encoder_graph.get_tensor_by_name('decoder/decoder_pred:0')
        
        data.reset_iterator('encoder')
        data.reset_iterator('decoder')
        encoder_x_batch, encoder_y_batch = data.next_batch('encoder', 'test')
        decoder_x_batch, decoder_y_batch = data.next_batch('decoder', 'test')
        
        init_state = None
        with open('./log/model/state.pkl', 'rb') as f:
            init_state = pickle.load(f) 
#        print(self.encoder_last_outputs)
        with tf.Session() as sess:
            encoder_saver.restore(sess, './log/model/encoder_decoder.ckpt')
            feed_dict = {encoder_inputs:encoder_x_batch,
                         decoder_inputs:decoder_x_batch,
                         decoder_outputs:decoder_y_batch,
                         initial_state_encoder:init_state}
            
            pred = sess.run(pred_decoder, feed_dict=feed_dict)
            
            pred_ = data.denormalize(pred, 'meanCPUUsage')
            actual = data.denormalize(decoder_y_batch, 'meanCPUUsage')
            
            loss_ = tf.reduce_mean(tf.squared_difference(pred_, actual))
            print('loss test = ', loss_.eval())
                       
            
            
            
config = Config('config.json')
model_config = config.get_model_config_encoder()
data_config = config.get_data_config()

data = Data(data_config)
data.series_to_supervised('encoder', model_config['model_default']['sliding_encoder'], 0)
data.series_to_supervised('decoder', model_config['model_default_encoder']['sliding_decoder'], 1)


model = EncoderDecoder(model_config)

#startTime = datetime.now()
#model.train_model(data)
#print('time training:', datetime.now() - startTime)
model.evaluate(data)


#            
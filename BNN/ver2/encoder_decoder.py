# -*- coding: utf-8 -*-
from datetime import datetime
start_time = datetime.now()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def multi_lstm(num_layers, num_units, keep_prob=1):
    cells = []
    for i in range(num_layers):
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)

def plot_loss(filename, train_losses, val_losses=None):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='train loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.savefig('./log/figure/' + filename)
    
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n...t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1,...t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with Nan values
    if(dropnan):
        agg.dropna(inplace=True)
    return agg
    

tf.reset_default_graph()

encoder_sliding = 4
num_features = 1
decoder_sliding = 4
batch_size = 8
num_epochs = 100
num_layers = 2
num_units = 4
keep_prob = 0.8

# placeholder for input data    
data_encoder_x = tf.placeholder(dtype=tf.float32, shape=[None, encoder_sliding, num_features], name='data_encoder_x')
data_decoder_x = tf.placeholder(dtype=tf.float32, shape=[None, decoder_sliding, num_features], name='data_decoder_x')
data_decoder_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='data_decoder_y')

# create dataset of tensorflow
dataset_encoder = tf.data.Dataset.from_tensor_slices(data_encoder_x)
dataset_decoder = tf.data.Dataset.from_tensor_slices((data_decoder_x, data_decoder_y))

# initial batch_size
dataset_encoder = dataset_encoder.batch(batch_size)
dataset_decoder = dataset_decoder.batch(batch_size)

# initial repeat dataset for epochs
dataset_encoder = dataset_encoder.repeat(num_epochs)
dataset_decoder = dataset_decoder.repeat(num_epochs)

# make iterator
iterator_encoder = dataset_encoder.make_initializable_iterator()
iterator_decoder = dataset_decoder.make_initializable_iterator()

## get values from dataset
#encoder_inputs = iterator_encoder.get_next()
#decoder_inputs, decoder_outputs = iterator_decoder.get_next()




# input for model
encoder_x = iterator_encoder.get_next()
decoder_x, decoder_y = iterator_decoder.get_next()

#encoder_x = tf.placeholder(dtype=tf.float32, shape=[None, encoder_sliding, num_features], name='encoder_x') # (batch_size, sequence_length, num_feature)
#decoder_x = tf.placeholder(dtype=tf.float32, shape=[None, decoder_sliding, num_features], name='decoder_x')
#decoder_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='decoder_y')

init_state = tf.placeholder(dtype=tf.float32, shape=[num_layers, 2, batch_size, num_units], name='init_state')

# init weight / bias output
weight_out = tf.Variable(tf.random_normal([num_units, 1]), name='weight_out')
bias_out = tf.Variable(tf.random_normal([1]), name='bias')

# build model encoder
with tf.variable_scope('encoder'):
    # get tuple state for initialize encoder
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], 
                                           state_per_layer_list[idx][1])
            for idx in range(num_layers)]
            )
            
#        print(rnn_tuple_state)
    
    encoder_cell = multi_lstm(num_layers, num_units, keep_prob=keep_prob)
    
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, 
                                                       inputs=encoder_x,
                                                       initial_state=rnn_tuple_state,
                                                       dtype=tf.float32)

# build model decoder
with tf.variable_scope('decoder'):
    decoder_cell = multi_lstm(num_layers, num_units, keep_prob=keep_prob)
    
    decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, 
                                                       inputs=decoder_x,
                                                       initial_state=encoder_state)
    
    decoder_outputs_last = decoder_outputs[:, -1, :]
    pred_decoder = tf.matmul(decoder_outputs_last, weight_out) + bias_out

with tf.name_scope('loss_optimizer'):
    loss = tf.reduce_mean(tf.sqrt(tf.squared_difference(pred_decoder, decoder_y)))
    optimizer  = tf.train.AdamOptimizer().minimize(loss)



# prepare data
df = pd.read_csv('../data/10Minutes.csv', header=None, 
                 names=["time_stamp", "numberOfTaskIndex", "numberOfMachineId", 
             "meanCPUUsage", "canonical_memory_usage", "AssignMem",
             "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
             "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
             "max_disk_io_time", "cpi", "mai", "sampling_portion",
             "agg_type", "sampled_cpu_usage"])
                
df = df.loc[:, 'meanCPUUsage']


data_en = series_to_supervised(df.values.reshape(df.shape[0],1), encoder_sliding, 0)
data_de = series_to_supervised(df.values.reshape(df.shape[0],1), decoder_sliding, 1)

e_x = data_en.values.reshape((data_en.shape[0], data_en.shape[1], 1))
d_x = data_de.iloc[:, :decoder_sliding].values
d_x = d_x.reshape((d_x.shape[0], d_x.shape[1], 1))
d_y = data_de.iloc[:, -1].values
d_y = d_y.reshape((d_y.shape[0],1))
                



initialize = tf.global_variables_initializer()
_current_state = np.zeros((num_layers, 2, batch_size, num_units))


batches_per_epoch = int (len(e_x)/batch_size)


with tf.Session() as sess:
    sess.run(initialize)
#    writer = tf.summary.FileWriter('./log/', sess.graph)
    
    
    sess.run(iterator_encoder.initializer, feed_dict={data_encoder_x:e_x})
    
    sess.run(iterator_decoder.initializer, feed_dict={data_decoder_x:d_x,
                                                      data_decoder_y:d_y})
    

    train_losses = []
    
    print(train_losses)
    
    index = 0
    while True:
        index += 1
        try:
#            print('index=', index)
            _current_state, l, o = sess.run([encoder_state, loss, optimizer], feed_dict={init_state: _current_state})
            if index % batches_per_epoch == 0:
                train_losses.append(l)
        except tf.errors.InvalidArgumentError:
            print('finish epoch', (index/batches_per_epoch))
            continue
        except tf.errors.OutOfRangeError:
            print('finished')
            break
        
    plt.plot(range(num_epochs), train_losses, label='train loss')
    plt.legend()
    plt.show()
        
    
                




print(datetime.now() - start_time)
























#config = Config('config.json')

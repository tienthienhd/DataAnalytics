# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from config import Config

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
    
def split_data(data, val_size=0.2, test_size=0.2):   
        ntest = int(round(len(data) * (1 - test_size)))
        nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
     
        df_train = data.iloc[:nval]
        df_val = data.iloc[nval:ntest]
        df_test = data.iloc[ntest:]
        data = dict()
        data['train'] = df_train
        data['val'] = df_val
        data['test'] = df_test
        
        return data

class Data(object):
    def __init__(self, config):
        self.df = pd.read_csv(config['data']['datapath'], header=None, names=config['data']['columns_full'])
        self.df = self.df.loc[:, config['data']['columns_brief']]
#        print(self.data.columns.values)
        # get parameter
#        self.num_samples = self.df.shape[0]
        self.num_features = self.df.shape[1]
        
        # normalize data
#        self.data.plot()
        self._normalize()
#        self.data.plot()
        

        # set other parameter
        self.sliding = dict()
        
    
    
    def _normalize(self):
        self.min = dict()
        self.max = dict()
        tmp = dict()
        for col in self.df.columns.values:
            data_col = self.df.loc[:, col].values
            self.min[col] = np.amin(data_col)
            self.max[col] = np.amax(data_col)
            tmp[col] = (data_col - self.min[col]) / (self.max[col] - self.min[col])
            
        self.df = pd.DataFrame(tmp)
        
    def get_max_min(self, feature):
        return self.max[feature], self.min[feature]
    
    def denormalize(self, data, feature):
        min_ = self.min[feature]
        max_ = self.max[feature]
        
        tmp = data * (max_ - min_) + min_
        return tmp
    
    def prepare_data_inputs(self, encoder_sliding=1, decoder_sliding=1):
        self.sliding['encoder'] = encoder_sliding
        self.sliding['decoder'] = decoder_sliding
        
        data_encoder = series_to_supervised(self.df, encoder_sliding, 0)
        data_decoder = series_to_supervised(self.df, decoder_sliding, 1)
        
        data = pd.concat([data_encoder, data_decoder], axis=1)
        data.dropna(inplace=True)
        
        self.data = split_data(data)
        
#        self.encoder_x = data.iloc[:, :encoder_sliding].values
#        self.decoder_x = data.iloc[:, encoder_sliding:encoder_sliding+decoder_sliding].values
#        self.decoder_y = data.iloc[:, encoder_sliding+decoder_sliding:].values
        
    
    def get_data(self, dataset):
        data = self.data[dataset] # train, val or test
        data = data.astype(np.float32)
        
        index_encoder = self.sliding['encoder']
        index_decoder = self.sliding['encoder'] + self.sliding['decoder']
        
        encoder_x = data.iloc[:, :index_encoder].values
        encoder_x = encoder_x.reshape((encoder_x.shape[0], encoder_x.shape[1], self.num_features))
        
        decoder_x = data.iloc[:, index_encoder:index_decoder].values
        decoder_x = decoder_x.reshape((decoder_x.shape[0], decoder_x.shape[1], self.num_features))
        
        decoder_y = data.iloc[:, index_decoder:].values
        
        return encoder_x, decoder_x, decoder_y
    
#    
    
#    
#config = Config('config.json')
#data = Data(config.get_config())
#data.prepare_data_inputs(4, 2)
#
#train = data.get_data('train')
#val = data.get_data('val')
#
#dataset_train = tf.data.Dataset.from_tensor_slices(train)
#dataset_val = tf.data.Dataset.from_tensor_slices(val)
#dataset_train = dataset_train.batch(4)
#print(dataset_train.output_types)
#print(dataset_train.output_shapes)
#
#iterator = tf.data.Iterator.from_structure(dataset_train.output_types)
#
#x = iterator.get_next()
#
#training_init_op = iterator.make_initializer(dataset_train)
#validation_init_op = iterator.make_initializer(dataset_val)
#
#
#sess = tf.Session()
#
#while True:
#    sess.run(training_init_op)
#    train_ = sess.run(x)
#    
#    sess.run(validation_init_op)
#    val_ = sess.run(x)
#    break

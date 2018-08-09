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
        return df_train, df_val, df_test

class Data(object):
    def __init__(self, config):
        self.df = pd.read_csv(config['datapath'], header=None, names=config['columns_full'])
        self.df = self.df.loc[:, config['columns_brief']]
#        print(self.data.columns.values)
        # get parameter
        self.num_samples = self.df.shape[0]
        self.num_features = self.df.shape[1]
        
        # normalize data
#        self.data.plot()
        self.normalize()
#        self.data.plot()
        
        # transform data to supervised

        # set other parameter
        self.data = dict()
        self.num_in = dict()
        self.num_out = dict()
        self.iterators = dict()
        self.batch_sizes = dict()
    
    def series_to_supervised(self, type_using, num_in, num_out):
        self.num_in[type_using] = num_in
        self.num_out[type_using] = num_out
        data = series_to_supervised(self.df, num_in, num_out)
        train, val, test = split_data(data)
        self.data[type_using] = dict()
        self.data[type_using]['train'] = train
        self.data[type_using]['val'] = val
        self.data[type_using]['test'] = test
        
        
    def get_num_features(self):
        return self.num_features
    
    def normalize(self):
        self.min = dict()
        self.max = dict()
        tmp = dict()
        for col in self.df.columns.values:
            data_col = self.df.loc[:, col].values
            self.min[col] = np.amin(data_col)
            self.max[col] = np.amax(data_col)
            tmp[col] = (data_col - self.min[col]) / (self.max[col] - self.min[col])
            
        self.df = pd.DataFrame(tmp)
        
        
    def denormalize(self, data, feature):
        min_ = self.min[feature]
        max_ = self.max[feature]
        
        tmp = data * (max_ - min_) + min_
        return tmp
                
        
        
        
    def init_iterator(self, iterator, start_index=0, batch_size=1):
        if iterator not in self.iterators:
            self.iterators[iterator] = start_index
            self.batch_sizes[iterator] = batch_size
        else:
            self.iterators[iterator] = start_index
            self.batch_sizes[iterator] = batch_size
        
    def reset_iterator(self, iterator, start_index=0):
        self.iterators[iterator] = start_index
        
    def has_next(self, iterator, dataset):
        if self.iterators[iterator] + self.batch_sizes[iterator] >= self.data[iterator][dataset].shape[0]:
            return False
        return True
        
    
    def next_batch(self, iterator, dataset):
        if self.iterators[iterator] >= self.num_samples:
            return None
        
        batch_size = self.batch_sizes[iterator]
        index = self.iterators[iterator]
        batch = self.data[iterator][dataset].iloc[index: index + batch_size].values
        self.iterators[iterator] = index + batch_size
        
        num_var_x = self.num_in[iterator] * self.num_features
        x = batch[:, :num_var_x]
        x = np.reshape(x, [x.shape[0], self.num_in[iterator], self.num_features])
        y = batch[:, num_var_x:]
        return x, y
#    
#
##config = Config('config.json')
##model_config = config.get_model_config()
##data_config = config.get_data_config()
##data = Data(data_config)
##data.series_to_supervised('encoder', model_config['encoder_num_inputs'], model_config['encoder_num_outputs'])
##data.series_to_supervised('decoder', model_config['decoder_num_inputs'], model_config['decoder_num_outputs'])
##
##data.init_iterator('encoder', start_index=0, batch_size=4)
##data.init_iterator('decoder', start_index=4, batch_size=4)
##
##        
##a = data.next_batch('decoder', 'train')
        
    
    
#    
#class DataSet(object):
#    def __init__(self, config):
#        df = pd.read_csv(config['datapath'], header=None, names=config['columns_full'])
#        self.df = df.loc[:, config['columns_brief']]
#        
#        
#        
##        print(self.data.columns.values)
#        # get parameter
##        self.num_samples = self.df.shape[0]
##        self.num_features = self.df.shape[1]
#        
#        # normalize data
##        self.data.plot()
#        self.normalize()
##        self.data.plot()
#        
##         set other parameter
#        self.data = dict()
##        self.num_in = dict()
##        self.num_out = dict()
##        self.iterators = dict()
##        self.batch_sizes = dict()
#        pass
#    
#    def series_to_supervised(self, dataset, num_in=1, num_out=1):
##        self.num_in[dataset] = num_in
##        self.num_out[dataset] = num_out
#        data = series_to_supervised(self.df, num_in, num_out)
#        train, val, test = split_data(data)
#        self.data[dataset] = dict()
#        self.data[dataset]['train'] = tf.data.Dataset.from_tensor_slices(train)
#        self.data[dataset]['val'] = tf.data.Dataset.from_tensor_slices(val)
#        self.data[dataset]['test'] = tf.data.Dataset.from_tensor_slices(test)
#    
#    def normalize(self):
#        self.min = dict()
#        self.max = dict()
#        tmp = dict()
#        for col in self.df.columns.values:
#            data_col = self.df.loc[:, col].values
#            self.min[col] = np.amin(data_col)
#            self.max[col] = np.amax(data_col)
#            tmp[col] = (data_col - self.min[col]) / (self.max[col] - self.min[col])
#            
#        self.df = pd.DataFrame(tmp)
#    
#    def denormalize(self, data, feature):
#        min_ = self.min[feature]
#        max_ = self.max[feature]
#        
#        tmp = data * (max_ - min_) + min_
#        return tmp
#    
#    def make_iterator(self):
#        pass
#    
#    def next_patch(self):
#        pass
    
    
    
    
    
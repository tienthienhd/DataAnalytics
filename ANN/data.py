# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:47:05 2018

@author: HP Zbook 15
"""

import pandas as pd
import numpy as np
import math

class Data(object):
    def __init__ (self, data_path, *col_names):
        self.columns = ['time_stamp', 'numberOfTaskIndex', 'numberOfMachineId', 
                 'meanCPUUsage', 'canonical_memory_usage', 'AssignMem',
                 'unmapped_cache_usage', 'page_cache_usage', 'max_mem_usage',
                 'mean_diskIO_time', 'mean_local_disk_space', 'max_cpu_usage',
                 'max_disk_io_time', 'cpi', 'mai', 'sampling_portion',
                 'agg_type', 'sampled_cpu_usage']
        self.data = pd.read_csv(data_path, names=self.columns, header=None)
        
        self.data = self.data.loc[:, col_names]
        self.min = {}
        self.max = {}

        
        
    
    def normalize_data(self, *col_names):
        result = {}
        for col in col_names:
            data_col = self.data.loc[:, col].values
            self.min[col] = np.amin(data_col)
            self.max[col] = np.amax(data_col)
            result[col] = (data_col - self.min[col]) / self.max[col]
        return pd.DataFrame(result)
    
    def denormalize_data(self, data, col_name):
        result = data * (self.max[col_name] - self.min[col_name]) + self.min[col_name]
        return result
            
    
    def split_data(self, data, val_size=0.2, test_size=0.2):
        ntest = int(round(len(data) * (1 - test_size)))
        nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
     
        df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
     
        return df_train, df_val, df_test
    
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
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
    
#    def generate_batch(self, data, batch_size):
#        df = pd.DataFrame(data)
#        self.dataset = df.values
#        self.batch_size = batch_size
#        num_batchs = int(math.ceil(len(self.dataset) / batch_size))
#        self.index_batch = 0
#        
#    def next_batch(self):
#        if self.batch_size is None:
#            print('please generate_batch')
#            return None
#        i = self.index_batch * self.batch_size
#        self.index_batch += 1
#        data = self.dataset[i: i + self.batch_size]
##        print(data.shape)
#        return data
    
#data = Data('./data/data_resource_usage_10Minutes_6176858948.csv', 'meanCPUUsage', 'canonical_memory_usage')
#normalized = data.normalize_data('meanCPUUsage')
##normalized.plot()
##pd.DataFrame(data.denormalize_data(normalized, 'canonical_memory_usage')).plot()
#
#train, val, test = data.split_data(normalized)
#train = data.series_to_supervised(train)
#data.generate_batch(train, 4)
#batch = data.next_batch()
#batch = data.next_batch()
#
#print(batch)
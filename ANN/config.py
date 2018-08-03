# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 20:12:50 2018

@author: HP Zbook 15
"""
import json
import pandas as pd
import numpy as np
import itertools

class Config(object):
    
    def __init__(self):
        self.index_config = 0
        pass
    
    def combination(self, array):
        b = []
        if len(array) <= 1:
            b = array
        else:
            b = list(itertools.product(array[0], array[1]))
            for i in range(2, len(array)):
                b = list(itertools.product(b, array[i]))
                for j in range(len(b)):
                    c = list(b[j][0])
                    c.append(b[j][1])
                    b[j] = c
        return b
    
    def grid(self, arrays):
        # base on numpy.meshgrid()
        ndim = len(arrays)
        s0 = (1,) * ndim
        output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
                  for i, x in enumerate(arrays)]
        output = np.broadcast_arrays(*output, subok=True)
        output = np.asarray(output)
        output = output.T.reshape(-1, ndim)
        return output
        
    def get_config_from_json(self, json_file):
        
        with open(json_file, 'r') as config_file:
            self.config_dict = json.load(config_file)
            
        return self.config_dict
    
    def generate_config(self):
        self.columns = self.config_dict.keys()
        tmp = [self.config_dict[key] for key in self.columns]
#        self.body = self.grid(tmp)
        self.body = self.combination(tmp)
        data = pd.DataFrame(self.body)
#        print(data.head())
        data.columns=self.columns
        data.to_csv('./log/config/config.csv', index=None)
        return len(self.body)
#        
    def next_config(self, index=None):
        
        if index is not None:
            config_array = self.body[index]
            config = dict(zip(self.columns, config_array))
            return config
        
        config_array = self.body[self.index_config]
#        config_ = []
#        
#        for item in config_array:
#            try:
#               config_.append(int(item)) 
#            except:
#                try:
#                    config_.append(float(item))
#                except:
#                    config_.append(item)
        
        self.index_config += 1
        config = dict(zip(self.columns, config_array))
        return config
            
            
            
            
#config = Config()
#config.get_config_from_json('config.json')
#num_combinations = config.generate_config()
#c = config.next_config()



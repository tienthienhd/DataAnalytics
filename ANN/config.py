# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 20:12:50 2018

@author: HP Zbook 15
"""
import json
import pandas as pd
import numpy as np

class Config(object):
    
    def __init__(self):
        self.index_config = 0
        pass
    
    def grid(self, arrays):
        # base on numpy.meshgrid()
        ndim = len(arrays)
        s0 = (1,) * ndim
        output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
                  for i, x in enumerate(arrays)]
        output = np.broadcast_arrays(*output, subok=True)
        output = np.array(output)
        output = output.T.reshape(-1, ndim)
        return output
        
    def get_config_from_json(self, json_file):
        
        with open(json_file, 'r') as config_file:
            self.config_dict = json.load(config_file)
            
        return self.config_dict
    
    def generate_config(self):
        self.columns = self.config_dict.keys()
        tmp = [self.config_dict[key] for key in self.columns]
        self.body = self.grid(tmp)
        data = pd.DataFrame(self.body)
#        print(data.head())
        data.columns=self.columns
        data.to_csv('./log/config/config.csv', index=None)
        return len(self.body)
#        
    def next_config(self):
        config_array = self.body[self.index_config]
        self.index_config += 1
        config = dict(zip(self.columns, config_array))
        return config
            
            
            
            
#config = config()
#config.get_config_from_json('config.json')
#config.generate_config()
#config1 = config.next_config()
#config2 = config.next_config()



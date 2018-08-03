# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 02:14:02 2018

@author: HP Zbook 15
"""
import json
import pandas as pd
import numpy as np
import itertools



class Config(object):
    def __init__(self, filepath):
        self.json_file = filepath
        self.get_config_from_json()
#        self.generate_config()
        
    
    def get_config_from_json(self):
        with open(self.json_file, 'r') as config_file:
            config_dict = json.load(config_file)
            self.model_default = config_dict['model_default']
            self.model_tuning = config_dict['model_default']
            
            self.model_default_encoder = config_dict['model_default_encoder']
            self.model_tuning_encoder = config_dict['model_tuning_encoder']
            
            self.data_config = config_dict['data_config']
            
            self.model_default_mlp = config_dict['model_default_mlp']
            self.model_tuning_mlp = config_dict['model_tuning_mlp']
            
    def get_data_config(self):
        return self.data_config
    
    def get_model_config_encoder(self):
        config = dict()
        config['model_default'] = self.model_default
        config['model_default_encoder'] = self.model_default_encoder
        return config
    
    def get_model_config_mlp(self):
        config = dict()
        config['model_default'] = self.model_default
        config['model_default_mlp'] = self.model_default_mlp
        return config
    
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

            
    def generate_config(self):
        self.columns = self.config_dict.keys()
        tmp = [self.config_dict[key] for key in self.columns]
#        self.body = self.grid(tmp)
        self.body = self.combination(tmp)
#        data = pd.DataFrame(self.body)
##        print(data.head())
#        data.columns=self.columns
#        data.to_csv('./log/config/config.csv', index=None)
        return len(self.body)
    
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
    
    
#config = Config('config.json')
#a = config.next_config()

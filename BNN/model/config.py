# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:31:09 2018

@author: HP Zbook 15
"""

import json
import pandas as pd
import numpy as np
import itertools

def combination(array):
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

def generate(config_dict):
    columns = config_dict.keys()
    tmp = [config_dict[key] for key in columns]
    body = combination(tmp)
    data = pd.DataFrame(body)
    data.columns=columns
    return data
    
class Config(object):
    def __init__(self, filename=None):
        if filename:
            with open(filename, 'r') as f:
                config = json.load(f)
                self.encoder_decoder = config['encoder_decoder']
                self.mlp = config['mlp']
                self.data = config['data']
            
            
    def get_config_tuning(self, filename):
        with open(filename, 'r') as f:
            self.config_tuning = json.load(f)
        return self.config_tuning
             
            
            
    def generate_config(self, type_config):
        config = self.config_tuning[type_config]
        return generate(config)
        
        
        
        
#        config_data = config['data']
#        config_common = config['common']
#        config_encoder_decoder = config['encoder_decoder']
#        config_mlp = config['mlp']
        
#        config_dict = {**config_common, **config_encoder_decoder, **config_mlp}
        
        
#        df = generate(config_dict)
        
#        df_a = pd.DataFrame(a)
#        df_b = pd.DataFrame(b)
#        
#        df = pd.concat((df_a, df_b), axis=1)
        
#        return df
        
        pass
        
    
        
        
c = Config('config.json')
a = c.get_config_tuning('tuning_config.json')
df = c.generate_config('data')


#df.to_csv('tmp.csv')
#a = df.to_dict('records')
#print(type(a))

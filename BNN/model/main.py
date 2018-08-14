# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:25:20 2018

@author: HP Zbook 15
"""

from config import Config
from data import Data
from encoder_decoder import EncoderDecoder
from datetime import datetime
from mlp import MLP
import os

result_directory = './log/results/'

config = Config()
config.get_config_tuning('tuning_config.json')

df_config_data = config.generate_config('data')


train_encoder = True

for i in range(df_config_data.shape[0]):
    # configs of data
    data_config = dict(df_config_data.iloc[i, :])
    dir_result_data = result_directory + data_config['data_name'] + '/config_' + str(i) + '/'
    
    
    dataset = Data(data_config)
    
    if train_encoder:
        df_config_encoder_decoder = config.generate_config('encoder_decoder')
        dir_result_encoder = dir_result_data + 'encoder_decoder/'
        
        if not os.path.exists(dir_result_encoder):
            os.makedirs(dir_result_encoder)
        
        for j in range(df_config_encoder_decoder.shape[0]):
            
            # configs of encoder_decoder
            encoder_decoder_config = dict(df_config_encoder_decoder.iloc[i, :])
            encoder_decoder_config['num_features'] = len(data_config['features'])
            
            # create model of encoder decoder with config
            encoder_decoder = EncoderDecoder(encoder_decoder_config, dataset.get_max_min())
            
            # prepare data
            dataset.prepare_data_inputs_encoder(encoder_decoder_config['sliding'][0],
                                        encoder_decoder_config['sliding'][1])
            train = dataset.get_data_encoder('train')
            val = dataset.get_data_encoder('val')
            test = dataset.get_data_encoder('test')
            
            
            # train model encoder decoder
            encoder_decoder.fit(train, val, test, dir_result_encoder, 'config_'+str(j))
            
            
            
    else:
        df_config_mlp = config.generate_config('mlp')
        
        for j in range(df_config_mlp.shape[0]):
            mlp_config = dict(df_config_mlp.iloc[i, :])
            mlp_config['num_features'] = len(data_config['features'])
            
            mlp = MLP(mlp_config, dataset.get_max_min())
            
            # prepare data
            dataset.prepare_data_inputs_mlp(mlp_config['input_dim'])
            
            train = dataset.get_data_mlp('train')
            val = dataset.get_data_mlp('val')
            test = dataset.get_data_mlp('test')
            
            mlp.fit(train, val, test)
        
        

#
#config = Config('config.json')
#data = Data(config)
#data.prepare_data_inputs(config.encoder_decoder['sliding'][0], 
#                         config.encoder_decoder['sliding'][1])
#train = data.get_data('train')
#val = data.get_data('val')
#test = data.get_data('test')
#
#ed = EncoderDecoder(config, data.get_max_min())
#start_time = datetime.now()
#ed.fit(train, val, test)
#interval_time = datetime.now() - start_time
#print('time run encoder decoder', interval_time)
#
#train = data.get_data('train', False)
#val = data.get_data('train', False)
#test = data.get_data('test', False)
#
#mlp = MLP(config, data.get_max_min())
#mlp.fit(train, val, test)
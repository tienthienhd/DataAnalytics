# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:01:29 2018

@author: HP Zbook 15
"""

import multiprocessing as mp
import os
from encoder_decoder import EncoderDecoder
from config import Config
from data import Data


def train_multi_encoder(config, data_config, dataset, dir_result_data, pool):
    df_config_encoder_decoder = config.generate_config('encoder_decoder')
    dir_result_encoder = dir_result_data + 'encoder_decoder/'
    
    if not os.path.exists(dir_result_encoder):
        os.makedirs(dir_result_encoder)
    
#    results = []
    for j in range(2):
        
        pool.apply_async(train_encoder, 
                   args=(df_config_encoder_decoder, data_config, 
                         dataset, dir_result_encoder, j), callback=log_result)
#        results.append(result)
    
#    output = [p.get() for p in results]
#    print(results)
    

def train_encoder(df_config_encoder_decoder, data_config, dataset, 
                  dir_result_encoder, index_config):
    
    print('run config_', index_config)
    # configs of encoder_decoder
    encoder_decoder_config = dict(df_config_encoder_decoder.iloc[index_config, :])
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
    encoder_decoder.fit(train, val, test, dir_result_encoder, 'config_'+str(index_config))
    
    

def run(pool):    
    result_directory = './log/results/'
    config = Config()
    config.get_config_tuning('tuning_config.json')
    
    df_config_data = config.generate_config('data')
    
    is_train_encoder = True
    
    for i in range(df_config_data.shape[0]):
        # configs of data
        data_config = dict(df_config_data.iloc[i, :])
        dir_result_data = result_directory + data_config['data_name'] + '/config_' + str(i) + '/'
        
        
        dataset = Data(data_config)
    
        if is_train_encoder:
            train_multi_encoder(config, data_config, dataset, dir_result_data, pool)
            
        break

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)
    
if __name__ == '__main__':
    print('start')
    num_processes = mp.cpu_count()
    
    pool = mp.Pool(processes=num_processes)
    run(pool)
    pool.close()
    pool.join()
    
    







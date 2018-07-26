# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 00:05:51 2018

@author: HP Zbook 15
"""
import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from config import Config
from model import Model
from data import Data


def generate_x_y(data, n_in=1, n_out=1):
    x = data.iloc[:, 0:n_in].values
    x = np.reshape(x, [len(x), n_in])
    y = data.iloc[:, n_in:].values
    y = np.reshape(y, [len(y), n_out])
    return x, y

def write_log(log, test, filename):
    with open('./log/result/train/'+filename, 'w+') as f:
        df_log = pd.DataFrame(log)
        df_log.to_csv(f, index=None)
    with open('./log/result/test/'+filename, 'w+') as f:
        f.write('prediction,actual\n')
        for pred, actual in zip(test['prediction'], test['actual']):
            f.write('{},{}\n'.format(pred[0], actual[0]))
        
    with open('./log/result/test/loss_test.csv', 'a+') as f:
        f.write('%f\n' % test['loss_test'])
        
def plot_loss(loss_train, loss_val):
    epochs = range(len(loss_train))
    plt.plot(epochs, loss_train, label='training loss')
    if loss_val and len(loss_val) == len(epochs):
        plt.plot(epochs, loss_val, label='validation loss')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

#tf.reset_default_graph()
#sess = tf.Session()

try:
    os.remove('./log/result/test/loss_test.csv')
except OSError:
    pass


config = Config()
config.get_config_from_json('config.json')
num_combinations = config.generate_config()

#conf = config.next_config()


data = Data('./data/data_resource_usage_10Minutes_6176858948.csv', 'meanCPUUsage', 'canonical_memory_usage')
normalized = data.normalize_data('meanCPUUsage')

#supervised_data = data.series_to_supervised(normalized, n_in=conf['sliding'])
#train, val, test = data.split_data(supervised_data)
#
#
#
#x_train, y_train = generate_x_y(train, n_in=conf['sliding'], n_out=1)
#x_val, y_val = generate_x_y(val, n_in=conf['sliding'], n_out=1)
#x_test, y_test = generate_x_y(test, n_in=conf['sliding'], n_out=1)


#
#print(x_batch, y_batch)

#model = Model(sess, conf)
#sess.run(tf.global_variables_initializer())
#epochs, loss_train, loss_val = model.train_model(x_train, y_train, x_val, y_val)
#
#
#plt.plot(epochs, loss_train, label='training loss')
#if len(loss_val) == len(epochs):
#    plt.plot(epochs, loss_val, label='validation loss')
#
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.legend()
#plt.show()


    
for i in range(num_combinations):        
    tf.reset_default_graph()
    sess = tf.Session()
    
    conf = config.next_config()
    
    supervised_data = data.series_to_supervised(normalized, n_in=conf['sliding'])

    train, val, test = data.split_data(supervised_data)
    
    
    x_train, y_train = generate_x_y(train, n_in=conf['sliding'], n_out=1)
    x_val, y_val = generate_x_y(val, n_in=conf['sliding'], n_out=1)
    x_test, y_test = generate_x_y(test, n_in=conf['sliding'], n_out=1)
    
    y_test = data.denormalize_data(y_test, 'meanCPUUsage') # convert y_test to actual
    
    model = Model(sess, conf)
    
    sess.run(tf.global_variables_initializer())
    
#    writer = tf.summary.FileWriter('./log/', sess.graph)
#    break
    
    loss_train, loss_val = model.fit(x_train, y_train, x_val, y_val, verbose=0)
    
    y_pred = model.predict(x_test)
    y_pred = data.denormalize_data(y_pred[0], 'meanCPUUsage')
    
    loss_test = np.sqrt(np.mean(np.square(np.subtract(y_pred, y_test)))) #np.mean(np.abs(y_pred - y_test))
    
    test = {'prediction': y_pred, 'actual': y_test, 'loss_test': loss_test}
    
    
    log = {'loss_train': loss_train, 'loss_val':loss_val}
    write_log(log, test, 'config_{}.csv'.format(i))
    
    print('complete with config:', conf)
    
#    plot_loss(loss_train, loss_val)
    
    sess.close()
#    break

#write = tf.summary.FileWriter('./log', sess.graph)
#sess.close()

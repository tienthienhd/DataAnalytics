# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:38:42 2018

@author: HP Zbook 15
"""

import tensorflow as tf
import numpy as np
import json
#
#num_units = 4
#num_layers = 2
#batch_size = 4
#
#tf.reset_default_graph()
#
#x = tf.ones(shape=[4, 2, 1])
#
#
#
#cell = [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(num_layers)]
#
#cell = tf.nn.rnn_cell.MultiRNNCell(cells=cell)
#
#output, state = tf.nn.dynamic_rnn(cell, inputs=x, dtype=tf.float32)
#
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#print(sess.run(output))
#
#print()
#
#for i in state:
#    print(sess.run(i))
#    print()
#    
#sess.close()

#x = tf.get_variable('x', shape=[3], initializer=tf.zeros_initializer)
#y = tf.get_variable('y', shape=[4], initializer=tf.zeros_initializer)
#
#inc_x = x.assign(x+1)
#dec_y = y.assign(y-1)
#
#init = tf.global_variables_initializer()
#
#saver = tf.train.Saver()
#
#with tf.Session() as sess:
#    sess.run(init)
#    
#    inc_x.op.run()
#    dec_y.op.run()
#    
#    save_path = saver.save(sess, './log/model/model.ckpt')
#    print('Model saved in path: %s' % save_path)
#    

#tf.reset_default_graph()
#
#x = tf.placeholder(tf.float32, [1])
#y = tf.placeholder(tf.float32, [1])
##saver = tf.train.Saver()
#
#with tf.Session() as sess:
###    saver.restore(sess, './log/model/model.ckpt')
##    print('Model restored.')
##    print('x=', x.eval())
##    print('y=', y.eval())
#    
#    x_ = sess.run([x], feed_dict={x:[1]})
#    print(x_)
    


#a = np.zeros((2, 2, 4, 8))
#b = np.amax([1, 2,3])

# Creates a graph.
#from datetime import datetime
#
#device_name = 'cpu'  # Choose device from cmd line. Options: gpu or cpu
#shape = (7000, 7000)
#if device_name == "gpu":
#    device_name = "/gpu:0"
#else:
#    device_name = "/cpu:0"
#    
#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True
#
#
#with tf.device(device_name):
#    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
#    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
#    sum_operation = tf.reduce_sum(dot_operation)
#
#
#startTime = datetime.now()
#with tf.Session(config=config) as session:
#        result = session.run(sum_operation)
#        print(result)
#
## It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
#print("\n" * 5)
#print("Shape:", shape, "Device:", device_name)
#print("Time taken:", datetime.now() - startTime)
#
#print("\n" * 5)


sess = tf.Session()


dataset = tf.data.Dataset.range(100)

dataset = dataset.batch(6)
dataset = dataset.repeat(4)
iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  print(value)
  
  
sess.close()  
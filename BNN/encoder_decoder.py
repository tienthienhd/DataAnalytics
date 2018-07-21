import tensorflow as tf 
import numpy as np 

def lstm_cell(state_size):
    return tf.nn.rnn
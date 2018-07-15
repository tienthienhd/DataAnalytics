import pandas as pd 
import tensorflow as tf 
import numpy as np 

# preparing the data
dataset = pd.read_csv('F:\LabDataAnalytics\data\iris\iris.data', header=None)
dataset.columns= ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
print(dataset.head())

X_ = dataset.iloc[:, :4].values
y_ = dataset.iloc[:, 4].values

labels = {'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
y_ = [labels[x] for x in y_]
y_ = np.reshape(y_, (len(y_,)))



# split to training and testing set
num_train = int(len(X_) * 0.8)
print('number of train:', num_train)
X_train = X_[:num_train]
y_train = y_[:num_train]

X_test = X_[num_train:]
y_test = y_[num_train:]

# build model
x = tf.placeholder(dtype=tf.float32, name='x', shape=(None, 4))
y = tf.placeholder(dtype=tf.float32, name='y', shape=(None, ))


##### CIFAR-10 CNN #####     
# by Zane Warner
# This project is my first ever implementation of a CNN using TensorFlow
# Its structure loosely follows the instructions of the 2nd homework in Stanford's 2017 cs231n course
# It will classify images from the CIFAR-10 dataset
##########

##TODO
# Set up training and validation
# Tune hyperparams (batch size and learning rate)
# Investigate & consider adding batch norm, dropout, learning rate decay
# Also investigate going deeper - will likely involve some method of downsampling

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

##### CIFAR-10 #####
# The CIFAR-10 dataset is described in the report titled:
#    Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
#
# The following code snippet, provided by the CIFAR-10 website, is used to unpack the CIFAR-10 database:
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dataDict = pickle.load(fo, encoding='bytes')
    return dataDict
##########

##### Load Data #####
imageDataDicts = []

imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_1'))
imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_2'))
imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_3'))
imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_4'))

numPrebuiltBatches = 4
inputData = []
outputLabels = []
for i in range(numPrebuiltBatches):
    inputData.append(imageDataDicts[i][b'data'])
    outputLabels.append(imageDataDicts[i][b'labels'])

validationDataDict = unpickle('cifar-10-batches-py/data_batch_5')
validationInputData = validationDataDict[b'data']
validationOutputLabels = validationDataDict[b'labels']

##### Graph Creation #####
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
trainingMode = tf.placeholder(tf.bool) #True indicates that it is training, false is test time

def makeTwoConvLayersGraph(x):
    #This creates the graph for a network that goes conv-relu-conv-relu-affine
    #The output is not activated after the dense later since it is designed to be fed to a loss function (e.g. a softmax)
    #Initialize variables
    filter1 = tf.get_variable("filter1", [32,32,3,32])
    bias1 = tf.get_variable("bias1", [32])
    filter2 = tf.get_variable("filter2", [32,32,32,16])
    bias2 = tf.get_variable("bias2", [16])
    
    #Build Graph
    c1 = tf.nn.conv2d(x, filter1, strides=[1,1,1,1], padding="SAME", name="c1") + bias1
    a1 = tf.nn.relu(c1, name="a1")
    c2 = tf.nn.conv2d(a1, filter2, strides=[1,1,1,1], padding="SAME", name="c2") + bias2
    a2 = tf.nn.relu(c2, name="a2")
    fc3 = tf.layers.dense(a2, units=10, name="fc3") #note that this name will be made weird by the autoaddition of a bias node
    return fc3

outputLayer = makeTwoConvLayersGraph(x)
loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), outputLayer)

optimizer = tf.train.AdamOptimizer(1e-5)
train_step = optimizer.minimize(loss)

netSaver = tf.train.Saver()

##### Training #####


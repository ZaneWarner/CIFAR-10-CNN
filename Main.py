##### CIFAR-10 CNN #####     
# by Zane Warner
# This project is my first ever implementation of a CNN using TensorFlow
# Its structure loosely follows the instructions of the 2nd homework in Stanford's 2017 cs231n course
# It will classify images from the CIFAR-10 dataset
##########

##TODO
# Tune hyperparams (batch size and learning rate)
# Investigate & consider adding batch norm, dropout, learning rate decay
# Also investigate going deeper - will likely involve some method of downsampling

##Testing Notes
# First sweep of learning rates found optimal rate to be 1e(-5 +/- 1)

import numpy as np
import tensorflow as tf
import math
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

inputData = imageDataDicts[0][b'data']
outputLabels = np.array(imageDataDicts[0][b'labels'])
for i in [1,2,3]:
    np.append(inputData, imageDataDicts[i][b'data'], axis=0)
    np.append(outputLabels, np.array(imageDataDicts[i][b'labels']), axis=0)
    
inputData = inputData.reshape(-1, 32, 32, 3)

validationDataDict = unpickle('cifar-10-batches-py/data_batch_5')
validationInputData = validationDataDict[b'data']
validationOutputLabels = np.array(validationDataDict[b'labels'])

validationInputData = validationInputData.reshape(-1, 32, 32, 3)

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
    a2Flat = tf.reshape(a2, [-1, 32*32*16])
    fc3 = tf.layers.dense(a2Flat, units=10, name="fc3") #note that this name will be made weird by the autoaddition of a bias node
    return fc3

outputLayer = makeTwoConvLayersGraph(x)
loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), outputLayer)
numCorrect = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(outputLayer, axis=1), y), tf.float32))

optimizer = tf.train.AdamOptimizer(1e-5)
trainer = optimizer.minimize(loss)

netSaver = tf.train.Saver()

##### Running the Network #####
def runNN(session, xIn, yIn, trainer=None, epochs=1, batchSize=100, printEvery = 20, lossPlot=False, plotname='lossplot.png'):
    lossValues = []
    accuracies = []
    numExamples = len(xIn)
    numBatches = math.ceil(numExamples/batchSize)
    batchRemainder = numExamples % batchSize
    batchIndices = np.arange(numExamples)
    for epoch in range(epochs):
        np.random.shuffle(batchIndices)
        batchLossAggregator = 0
        correctAggregator = 0
        for batch in range(numBatches):
            batchLower = batch*batchSize
            if batch != numBatches:
                batchUpper = batchLower+batchSize
            else:
                batchUpper = batchLower+batchRemainder
            thisBatch = batchIndices[batchLower:batchUpper]
            values = {x : xIn[np.array(thisBatch)],
                      y : yIn[np.array(thisBatch)]}
            if trainer is not None:
                batchLossValue, correct, _ = session.run([loss, numCorrect, trainer], feed_dict = values)
            else:
                batchLossValue, correct = session.run([loss, numCorrect], feed_dict = values)
            batchLossAggregator += batchLossValue
            correctAggregator += correct
        lossValues.append(batchLossAggregator)
        accuracies.append(correctAggregator/numExamples)
        if epoch % printEvery == 0 and trainer is not None:
            print('Training Epoch {}--Loss Value: {}, Accuracy: {}'.format(epoch+1, lossValues[epoch], accuracies[epoch]))
        elif trainer is None: print('Test Outcomes--Loss Value: {}, Accuracy: {}'.format(lossValues[epoch], accuracies[epoch]))
    
    netSaver.save(session, './NetworkSaves/CNN')   
    if lossPlot == True:
        plt.plot(lossValues)
        plt.savefig(plotname)
        plt.clf()
    
    return lossValues, accuracies

#### Execution #####
# with tf.Session() as sess:
#     with tf.device("/cpu:0"):
#         sess.run(tf.global_variables_initializer())
#         print('Training')
#         runNN(sess, inputData[:100,:], outputLabels[:100,], trainer=trainer, batchSize=50, epochs=2, printEvery=2, lossPlot=True)
        #print('Validation')
        #runNN(sess, validationInputData, validationOutputLabels, trainer=None, batchSize=10, lossPlot=True)

testBatchSizes = [10, 25, 50, 100, 200, 400, 800, 2000]
for batchTest in testBatchSizes:
    testAccuracies = []
    #lrUse = 10**(-lrTest)
    #print('learning rate: {}'.format(lrUse))
    #optimizer = tf.train.AdamOptimizer(lrUse)
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(tf.global_variables_initializer())
            runNN(sess, inputData, outputLabels, trainer=trainer, batchSize=batchTest, epochs=40, printEvery=5, lossPlot=False)
            print('Validation for Batch Size {}'.format(batchTest))
            testLossVals, testAccuracyVals = runNN(sess, validationInputData, validationOutputLabels, batchSize=batchTest, trainer=None, lossPlot=False, printEvery=5)
            testAccuracies.append(testAccuracyVals[0])

plt.plot(testBatchSizes, testAccuracies)
plt.savefig('batchSizeTestResults')
plt.clf()

##### CIFAR-10 CNN #####     
# by Zane Warner
# This project is my first ever implementation of a CNN using TensorFlow
# Its structure loosely follows the instructions of the 2nd homework in Stanford's 2017 cs231n course
# It will classify images from the CIFAR-10 dataset
##########

##TODO
# Random image cropping sounds like a pain to implement but could also help with overfitting
# Tune hyperparams (batch size and learning rate, regularization beta and dropout probability)

##Testing Notes
# First sweep of learning rates found optimal rate to be 1e(-5 +/- 1)
# First batch size sweep was inconclusive but found extensive overfitting even at small batch sizes.
# Adding Batch Normalizaiton helped reduce the degree of overfitting, but insufficiently
# Adding max-pool dropout helped reduce the degree of overfitting, but insufficiently
# Adding horizontal image flips with minor random hue shift only provided negligible improvement in the overfitting problem
# Adding L2 regularization significantly helped the overfitting problem
# Adding a third conv layer to reduce the number of filters helped the overfitting problem
# Re-sweeping learning rates and batch sizes in a lattice found best results with learning rate around 1e-5 (bounded by worse performance at 5e-5 and 5e-6, with 5e-6 much closer)
#     and batch size of 100 (bounded by worse performance at 50 and 200)
#Testing dropout probabilities found .5 to be the optimal (with bounds of .45 and .55 on either side)

import numpy as np
import tensorflow as tf
import math
from DataAugmenter import HorizontalFlip, RandomHueShift
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
    
inputData = np.float64(inputData.reshape(-1, 32, 32, 3))
channel1Mean = np.mean(inputData[:,:,:,0])
channel2Mean = np.mean(inputData[:,:,:,1])
channel3Mean = np.mean(inputData[:,:,:,2])
inputData[:,:,:,0] -= channel1Mean
inputData[:,:,:,1] -= channel2Mean
inputData[:,:,:,2] -= channel3Mean
 
augmentedData = np.zeros(inputData.shape)
for i in range(len(inputData[:,0,0,0])):
    img = inputData[i,:,:,:]
    newImg = RandomHueShift(HorizontalFlip(img))
    augmentedData[i,:,:,:] = newImg
inputData = np.append(inputData, augmentedData, 0)
outputLabels = np.append(outputLabels, outputLabels, 0)
 
validationDataDict = unpickle('cifar-10-batches-py/data_batch_5')
validationInputData = validationDataDict[b'data']
validationOutputLabels = np.array(validationDataDict[b'labels'])
 
validationInputData = np.float64(validationInputData.reshape(-1, 32, 32, 3))
validationInputData[:,:,:,0] -= channel1Mean
validationInputData[:,:,:,1] -= channel2Mean
validationInputData[:,:,:,2] -= channel3Mean

##### Graph Creation #####
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
trainingMode = tf.placeholder(tf.bool) #True indicates that it is training, false is test time

def makeTwoConvLayersGraph(x, dropoutProb=.5, betaConv=100, betaFC=.1):
    #This creates the graph for a network that goes conv-relu-conv-relu-affine
    #The output is not activated after the dense later since it is designed to be fed to a loss function (e.g. a softmax)
    #Initialize variables
    filter1 = tf.get_variable("filter1", [32,32,3,16])
    bias1 = tf.get_variable("bias1", [16])
    filter2 = tf.get_variable("filter2", [16,16,16,16])
    bias2 = tf.get_variable("bias2", [16])
    filter3 = tf.get_variable("filter3", [16,16,16,8])
    bias3 = tf.get_variable("bias3", [8])
    fcWeights4 = tf.get_variable("fcWeights4", [512, 1000])
    fcBias4 = tf.get_variable("fcBias4", [1000])
    
    #Build Graph
    c1 = tf.nn.conv2d(x, filter1, strides=[1,1,1,1], padding="SAME", name="c1") + bias1
    a1 = tf.nn.relu(c1, name="a1")
    bn1 = tf.layers.batch_normalization(a1, axis=3, training=trainingMode, name="bn1")
    drpo1 = tf.layers.dropout(bn1, rate=dropoutProb, name="drpo1")
    mp1 = tf.nn.max_pool(drpo1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name='mp1')
    c2 = tf.nn.conv2d(mp1, filter2, strides=[1,1,1,1], padding="SAME", name="c2") + bias2
    a2 = tf.nn.relu(c2, name="a2")
    bn2 = tf.layers.batch_normalization(a2, axis=3, training=trainingMode, name="bn2")
    drpo2 = tf.layers.dropout(bn2, rate=dropoutProb, name="drpo2")
    mp2 = tf.nn.max_pool(drpo2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name='mp2')
    c3 = tf.nn.conv2d(mp2, filter3, strides=[1,1,1,1], padding="SAME", name="c3") + bias3
    a3 = tf.nn.relu(c3, name="a3")
    bn3 = tf.layers.batch_normalization(a3, axis=3, training=trainingMode, name="bn3")
    bn3Flat = tf.reshape(bn3, [-1, 8*8*8])
    fc4 = tf.matmul(bn3Flat, fcWeights4) + fcBias4
    a4 = tf.nn.relu(fc4, name="a4")
    bn4 = tf.layers.batch_normalization(a4, training=trainingMode, name="bn4")
    drpo4 = tf.layers.dropout(bn4, rate=dropoutProb, name="drpo4")
    fc5 = tf.layers.dense(drpo4, 10, name="fc5")
    
    convRegularizer = betaConv*(tf.nn.l2_loss(filter1, name="filter1Reg") + tf.nn.l2_loss(filter2, name="filter2Reg") + tf.nn.l2_loss(filter3, name="filter3Reg"))
    fcRegularizer =  betaFC*tf.nn.l2_loss(fcWeights4, name="fcWeights4Reg")
    l2Regularizer = convRegularizer + fcRegularizer
    return fc5, l2Regularizer

outputLayer, regularizer  = makeTwoConvLayersGraph(x)
loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), outputLayer) + regularizer
numCorrect = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(outputLayer, axis=1), y), tf.float32))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

optimizer = tf.train.AdamOptimizer(1e-5)
with tf.control_dependencies(update_ops):
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
            if trainer is not None:
                values = {x : xIn[np.array(thisBatch)],
                          y : yIn[np.array(thisBatch)],
                          trainingMode : True}
                batchLossValue, correct, _ = session.run([loss, numCorrect, trainer], feed_dict = values)
            else:
                values = {x : xIn[np.array(thisBatch)],
                          y : yIn[np.array(thisBatch)],
                          trainingMode : False}
                batchLossValue, correct = session.run([loss, numCorrect], feed_dict = values)
            batchLossAggregator += batchLossValue*len(thisBatch)
            correctAggregator += correct
        lossValues.append(batchLossAggregator/numExamples)
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
with tf.Session() as sess:
    with tf.device("/gpu:0"):
        sess.run(tf.global_variables_initializer())
        print('Training')
        runNN(sess, inputData, outputLabels, trainer=trainer, epochs=65, printEvery=5, lossPlot=False)
        print('Validation')
        runNN(sess, validationInputData, validationOutputLabels, trainer=None, batchSize=1000, lossPlot=False)

# testBatchSizes = [10, 25, 50, 100, 200, 400, 800, 2000]
# testLearningRates = [1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7, 1e-8,]
# testAccuracies = np.zeros([8,10])
# for i in range(len(testBatchSizes)):
#     for j in range(len(testLearningRates)):
#         optimizer = tf.train.AdamOptimizer(testLearningRates[j])
#         with tf.control_dependencies(update_ops):
#             trainer = optimizer.minimize(loss)
#         with tf.Session() as sess:
#             with tf.device("/gpu:0"):
#                 sess.run(tf.global_variables_initializer())
#                 runNN(sess, inputData, outputLabels, trainer=trainer, batchSize=testBatchSizes[i], epochs=65, printEvery=5, lossPlot=False)
#                 print('Validation for Batch Size {}, Learning Rate {}'.format(testBatchSizes[i], testLearningRates[j]))
#                 testLossVals, testAccuracyVals = runNN(sess, validationInputData, validationOutputLabels, batchSize=1000, trainer=None, lossPlot=False, printEvery=5)
#                 testAccuracies[i, j] = testAccuracyVals[0]
#   
# print("Test Accuracy Matrix (Rows by Batch Size, Cols by Learning Rate):")
# print(testAccuracies)
# topAccuracyIndex = np.argmax(testAccuracies)
# topAccuracyIndexBS = topAccuracyIndex // 10
# topAccuracyIndexLR = topAccuracyIndex % 10
# print("Highest Test Accuracy Obtained at Indices {},{}".format(topAccuracyIndexBS, topAccuracyIndexLR))

##### CIFAR-10 CNN #####     
# by Zane Warner
# This project is my first ever implementation of a CNN using TensorFlow
# Its structure loosely follows the instructions of the 2nd homework in Stanford's 2017 cs231n course
# It will classify images from the CIFAR-10 dataset
##########

##TODO
# Random image cropping is spooky in testing phase, rescaling to avoid stochasitcity could be better but is hard
# Tune hyperparams (batch size and learning rate, regularization beta and dropout probability)
#   LR might be too high, testing with betas .1/.01 plateau'd at epoch 41 (could also be overregularization)

##Testing Notes
# New deeper architecture
# With LR 5e-3 and BS 100, 151 epochs:
#   B1=0, B2=0: .8766 / .5254, training acc continuing to improve w training
#   B1=100, B2=0: .3623 / .199, training plateau v quick
#   B1=10, B2=0: .4293 / .1595, training plateau around 111
#   B1=1, B2=0: .5107 / .4089
#   B1=.5 B2=0, .5713 / .4058
#   B1=.1 B2=0, .7128 / .4407
#   B1=.1 B2=.1, .4136 / .3585
#   B1=.1 B2=.01, .4265 / .3592



import numpy as np
import tensorflow as tf
import math
from DataAugmenter import HorizontalFlip, RandomHueShift, RandomHorizontalFlip, RandomCropper
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
    
# Adjust Channel Means To 0 
inputData = np.float64(inputData.reshape(-1, 32, 32, 3))
channel1Mean = np.mean(inputData[:,:,:,0])
channel2Mean = np.mean(inputData[:,:,:,1])
channel3Mean = np.mean(inputData[:,:,:,2])
inputData[:,:,:,0] -= channel1Mean
inputData[:,:,:,1] -= channel2Mean
inputData[:,:,:,2] -= channel3Mean
 
#Load Validation
validationDataDict = unpickle('cifar-10-batches-py/data_batch_5')
validationInputData = validationDataDict[b'data']
validationOutputLabels = np.array(validationDataDict[b'labels'])

# Adjust Validation Channel Means To 0 
validationInputData = np.float64(validationInputData.reshape(-1, 32, 32, 3))
validationInputData[:,:,:,0] -= channel1Mean
validationInputData[:,:,:,1] -= channel2Mean
validationInputData[:,:,:,2] -= channel3Mean

##### Graph Creation #####
x = tf.placeholder(tf.float32, [None, 28, 28, 3]) #The 28s depend on the amount of random cropping used. In particular, they should be 32 minus the amount cropped.
y = tf.placeholder(tf.int64, [None])
trainingMode = tf.placeholder(tf.bool) #True indicates that it is training, false is test time

def makeTwoConvLayersGraph(x, dropoutProb=.5, betaConv=.1, betaFC=.01):
    #This creates the graph for a network that goes conv-relu-conv-relu-affine
    #The output is not activated after the dense later since it is designed to be fed to a loss function (e.g. a softmax)
    #Initialize variables
    filter1 = tf.get_variable("filter1", [2,2,3,32])
    bias1 = tf.get_variable("bias1", [32])
    filter2 = tf.get_variable("filter2", [2,2,32,16])
    bias2 = tf.get_variable("bias2", [16])
    filter3 = tf.get_variable("filter3", [2,2,16,8])
    bias3 = tf.get_variable("bias3", [8])
    filter4 = tf.get_variable("filter4", [2,2,8,8])
    bias4 = tf.get_variable("bias4", [8])
    fcWeights5 = tf.get_variable("fcWeights5", [7*7*8, 1000]) #these dimensions depend on the amount of random cropping
    fcBias5 = tf.get_variable("fcBias5", [1000])
    fcWeights6 = tf.get_variable("fcWeights6", [1000, 1000])
    fcBias6 = tf.get_variable("fcBias6", [1000])
    
    #Build Graph
    # conv-relu-bn-dropout-maxpool 1
    c1 = tf.nn.conv2d(x, filter1, strides=[1,1,1,1], padding="SAME", name="c1") + bias1
    a1 = tf.nn.relu(c1, name="a1")
    bn1 = tf.layers.batch_normalization(a1, axis=3, training=trainingMode, name="bn1")
    drpo1 = tf.layers.dropout(bn1, rate=dropoutProb, name="drpo1")
    mp1 = tf.nn.max_pool(drpo1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name='mp1')
    # conv-relu-bn-dropout-maxpool 2
    c2 = tf.nn.conv2d(mp1, filter2, strides=[1,1,1,1], padding="SAME", name="c2") + bias2
    a2 = tf.nn.relu(c2, name="a2")
    bn2 = tf.layers.batch_normalization(a2, axis=3, training=trainingMode, name="bn2")
    drpo2 = tf.layers.dropout(bn2, rate=dropoutProb, name="drpo2")
    mp2 = tf.nn.max_pool(drpo2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name='mp2')
    # conv-relu-bn
    c3 = tf.nn.conv2d(mp2, filter3, strides=[1,1,1,1], padding="SAME", name="c3") + bias3
    a3 = tf.nn.relu(c3, name="a3")
    bn3 = tf.layers.batch_normalization(a3, axis=3, training=trainingMode, name="bn3")
    # conv-relu-bn
    c4 = tf.nn.conv2d(bn3, filter4, strides=[1,1,1,1], padding="SAME", name="c4") + bias4
    a4 = tf.nn.relu(c4, name="a4")
    bn4 = tf.layers.batch_normalization(a4, axis=3, training=trainingMode, name="bn4")
    bn4Flat = tf.reshape(bn4, [-1, 7*7*8]) #these dimensions depend on the amount of random cropping
    # fc-relu-bn-dropout
    fc5 = tf.matmul(bn4Flat, fcWeights5) + fcBias5 #A 1000 hidden unit layer, done manually to make regularization more straightforward
    a5 = tf.nn.relu(fc5, name="a5")
    bn5 = tf.layers.batch_normalization(a5, training=trainingMode, name="bn5")
    drpo5 = tf.layers.dropout(bn5, rate=dropoutProb, name="drpo5")
    # fc-relu-bn-dropout
    fc6 = tf.matmul(drpo5, fcWeights6) + fcBias6 #A 1000 hidden unit layer, done manually to make regularization more straightforward
    a6 = tf.nn.relu(fc6, name="a6")
    bn6 = tf.layers.batch_normalization(a6, training=trainingMode, name="bn6")
    drpo6 = tf.layers.dropout(bn6, rate=dropoutProb, name="drpo6")
    # fc output
    fc7 = tf.layers.dense(drpo6, 10, name="fc7")
    
    #regularization
    convRegularizer = betaConv*(tf.nn.l2_loss(filter1, name="filter1Reg") + tf.nn.l2_loss(filter2, name="filter2Reg") + tf.nn.l2_loss(filter3, name="filter3Reg") + tf.nn.l2_loss(filter4, name="filter4Reg"))
    fcRegularizer =  betaFC*(tf.nn.l2_loss(fcWeights5, name="fcWeights5Reg") + tf.nn.l2_loss(fcWeights6, name="fcWeights6Reg"))
    l2Regularizer = convRegularizer + fcRegularizer
    return fc7, l2Regularizer

outputLayer, regularizer  = makeTwoConvLayersGraph(x)
loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), outputLayer) + regularizer
numCorrect = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(outputLayer, axis=1), y), tf.float32))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

optimizer = tf.train.AdamOptimizer(5e-3)
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
        #data augmentation
        xAugmented = np.zeros([numExamples, 28, 28, 3]) #The 28s are yet another place that the amount of cropping is hardcoded.
        if trainer is not None:
            for i in range(numExamples):
                xAugmented[i, :, :, :] = RandomHueShift(RandomHorizontalFlip(RandomCropper(xIn[i, :, :, :], cropPixels=4)))
        else:
            for i in range(numExamples):
                xAugmented[i, :, :, :] = RandomCropper(xIn[i, :, :, :], cropPixels=4) #stochasticity during test time??????? this seems bad but idk how else to make the images fit
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
                values = {x : xAugmented[np.array(thisBatch)],
                          y : yIn[np.array(thisBatch)],
                          trainingMode : True}
                batchLossValue, correct, _ = session.run([loss, numCorrect, trainer], feed_dict = values)
            else:
                values = {x : xAugmented[np.array(thisBatch)],
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
        runNN(sess, inputData, outputLabels, trainer=trainer, epochs=151, printEvery=10, lossPlot=False)
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

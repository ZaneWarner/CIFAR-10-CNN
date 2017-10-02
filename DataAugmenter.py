##### CIFAR-10 CNN #####     
# by Zane Warner
# This project is my first ever implementation of a CNN using TensorFlow
# Its structure loosely follows the instructions of the 2nd homework in Stanford's 2017 cs231n course
# It will classify images from the CIFAR-10 dataset
##########

##### Data Augmenter #####
# this module contains functions that are used to augment the training data to combat overfitting/increase mdoel generalization
# they are designed to be called in the main module
# input images are assumed to be numpy arrays with dimensions 32x32x3, which is width x height x channels
##########

import numpy as np

def HorizontalFlip(img):
    flippedImg = np.zeros([32, 32, 3])
    for i in range(32):
        flippedImg[:,i,:] = img[:,31-i,:]
    return flippedImg
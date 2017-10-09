##### CIFAR-10 CNN #####     
# by Zane Warner
# This project is my first ever implementation of a CNN using TensorFlow
# Its structure loosely follows the instructions of the 2nd homework in Stanford's 2017 cs231n course
# It will classify images from the CIFAR-10 dataset
##########

##### Data Augmenter #####
# this module contains functions that are used to augment the training data to combat overfitting/increase mdoel generalization
# they are designed to be called in the main module
# images are assumed to be square with 3 color channels
##########

import numpy as np

def HorizontalFlip(img):
    flippedImg = np.zeros(img.shape)
    imgWidth = len(img[0,:,0])
    for i in range(imgWidth):
        flippedImg[:,i,:] = img[:,imgWidth-(i+1),:]
    return flippedImg

def RandomHorizontalFlip(img, flipProbability=.5):
    probabilitySample = np.random.random_sample()
    if probabilitySample < flipProbability:
        img = HorizontalFlip(img)
    return(img)

def RandomHueShift(img, hueFloor=.95, hueCeil=1.05):
    shiftedImg = np.zeros(img.shape)
    shiftAmts = np.random.uniform(hueFloor, hueCeil, 3)
    for i in range(3):
        shiftedImg[:,:,i] = img[:,:,i]*shiftAmts[i]
    return shiftedImg
        
def RandomCropper(img, cropPixels=4):
    imgSize = len(img[:,0,0])
    topBuffer = np.random.randint(cropPixels+1) #+1 because randint uses a half-open interval (upper half open)
    leftBuffer = np.random.randint(cropPixels+1)
    croppedImgSize = imgSize-cropPixels
    croppedImg = img[topBuffer:(croppedImgSize+topBuffer), leftBuffer:(croppedImgSize+leftBuffer), :]
    return croppedImg
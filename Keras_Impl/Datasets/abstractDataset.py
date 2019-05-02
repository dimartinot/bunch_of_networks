# CIFAR10 small image classification
# 
# Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
from keras.datasets import cifar10
from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self,value):
        self.value = value
        super().__init__()

    @abstractmethod
    def getTrainingData(self):
        """
            Method to overwrite to retrieve training data
        """

    @abstractmethod
    def getTestingData(self):
        """
            Method to overwrite to retrieve testing data
        """

    @abstractmethod
    def getInputShape(self):
        """
            Method to overwrite to retrieve the input shape to setup our network with.
        """

    @abstractmethod
    def getNumClasses(self):
        """
            Method to overwrite to retrieve the number of classes to setup our network with.
        """
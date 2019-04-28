import Keras_Impl.Datasets.abstractDataset as ds
import numpy as np
# CIFAR10 small image classification
#
# Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
from keras.datasets import cifar10

class Cifar10Dataset(ds.Dataset):

    def __init__(self):
        super().__init__(self)
        print("\tChosen Dataset: Cifar10Dataset")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
    
    def getTrainingData(self):
        return (self.x_train, self.y_train)

    def getTestingData(self):
        return (self.x_test, self.y_test)
    
    def flattenData(self):
        print(self.x_train.shape, self.y_train.shape)
        np.reshape(self.x_train, (50000,1024))
        np.reshape(self.y_train, (10000,1024))
        print(self.x_train.shape, self.y_train.shape)
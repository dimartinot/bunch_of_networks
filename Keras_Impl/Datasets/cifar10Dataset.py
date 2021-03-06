import Keras_Impl.Datasets.abstractDataset as ds
import numpy as np
# CIFAR10 small image classification
#
# Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
from keras.datasets import cifar10
from keras.utils import to_categorical

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
        print(self.x_train.shape, self.x_test.shape, self.y_train.shape)
        self.x_train = np.reshape(self.x_train, (50000,1024))
        self.x_test = np.reshape(self.x_test, (10000,1024))

        # Normalization

        self.x_train = np.array(self.x_train) / 255
        self.x_test = np.array(self.x_test) / 255

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

        print(self.x_train.shape, self.x_test.shape)

    def getInputShape(self):
        return 1024

    def getNumClasses(self):
        return 10
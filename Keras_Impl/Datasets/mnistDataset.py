import Keras_Impl.Datasets.abstractDataset as ds
import numpy as np
# MNIST database of handwritten digits
#
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
from keras.datasets import mnist
from keras.utils import to_categorical

class MNISTDataset(ds.Dataset):

    def __init__(self):
        super().__init__(self)
        print("\tChosen Dataset: MNISTDataset")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
    
    def getTrainingData(self):
        return (self.x_train, self.y_train)

    def getTestingData(self):
        return (self.x_test, self.y_test)
    
    def flattenData(self):
        print(self.x_train.shape, self.x_test.shape, self.y_train.shape)
        self.x_train = np.reshape(self.x_train, (60000,784))
        self.x_test = np.reshape(self.x_test, (10000,784))

        # Normalization

        self.x_train = np.array(self.x_train) / 255
        self.x_test = np.array(self.x_test) / 255

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

        print(self.x_train.shape, self.x_test.shape)

    def getInputShape(self):
        return 784

    def getNumClasses(self):
        return 10
import numpy as np
import Keras_Impl.Datasets.abstractDataset as ds
import keras

class DummyDataset(ds.Dataset):

    def __init__(self):
        super().__init__(self)
        # Generate dummy data
        print("\tChosen Dataset: DummyDataset")
        self.x_train = np.random.random((1000, 1024))
        self.y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
        self.x_test = np.random.random((100, 1024))
        self.y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    
    def getTrainingData(self):
        return (self.x_train, self.y_train)

    def getTestingData(self):
        return (self.x_test, self.y_test)
    
    def getInputShape(self):
        return 1024
    
    def getNumClasses(self):
        return 10


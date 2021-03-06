import numpy as np
import Keras_Impl.Datasets.abstractDataset as ds
import keras

class DummyDataset(ds.Dataset):

    def __init__(self, input_shape_desired, num_classes_desired, num_of_outputs=1):
        super().__init__(self)
        # Generate dummy data
        print("\tChosen Dataset: DummyDataset")
        self.input_shape = input_shape_desired
        self.num_classes = num_classes_desired

        if isinstance(input_shape_desired, tuple):
            shape_x_train = ((1000,), input_shape_desired)
            shape_x_train = sum(shape_x_train, ()) # flatten the possible "tuples in tuples" architecture
            shape_x_test = ((100,), input_shape_desired)
            shape_x_test = sum(shape_x_test, ()) # flatten the possible "tuples in tuples" architecture

        else:
            shape_x_train = (1000, input_shape_desired)
            shape_x_test = (100, input_shape_desired)
       
        self.x_train = np.random.random_sample(shape_x_train)
        self.y_train = keras.utils.to_categorical(np.random.randint(num_classes_desired, size=(1000, 1)), num_classes=num_classes_desired)
        self.x_test = np.random.random_sample(shape_x_test)
        self.y_test = keras.utils.to_categorical(np.random.randint(num_classes_desired, size=(100, 1)), num_classes=num_classes_desired)
        self.num_of_outputs = num_of_outputs
        
        if (num_of_outputs != 1):
            # we create the array of outputs if more than 1 output is needed
            self.y_train = np.repeat([self.y_train], [num_of_outputs], axis=0)
            self.y_test = np.repeat([self.y_test], [num_of_outputs], axis=0)
    
    def getTrainingData(self):
        return (self.x_train, self.y_train)

    def getTestingData(self):
        return (self.x_test, self.y_test)
    
    def getInputShape(self):
        return self.input_shape
    
    def getNumClasses(self):
        return self.num_classes

    def getNumberOfOutputs(self):
        return self.num_of_outputs


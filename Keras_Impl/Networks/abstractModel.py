from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def createModel(self):
        """
            Function to overwrite in order to create a model
        """
    
    @abstractmethod
    def trainModel(self,train_dataset):
        """
            Function to overwrite in order to train the model
        """
    
    @abstractmethod
    def evaluateModel(self,test_dataset):
        """
            Function to overwrite to test the model
        """
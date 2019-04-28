from Keras_Impl.Networks.MLP import MLPModel as MLP
from Keras_Impl.Datasets import cifar10Dataset
from Keras_Impl.Datasets import dummyDataset

from termcolor import colored

def exec():
    print(colored('------- KERAS MENU -------','green'))
    print('Please select the model you would like to train and, maybe, test:')
    print('\n1°/ MLP')
    print('')
    try:
        choice = int(input("Selection > "))
    except:
        print("The choice has to be an integer... Default selection = 1")
        choice = 1
    if (choice == 1):
        model = MLP.MLPModel()
        print(colored("Creating the model...",'blue'))
        model.createModel()
        print("Loading the Dataset...")
        # ds = cifar10Dataset.Cifar10Dataset()
        # ds.flattenData()
        ds = dummyDataset.DummyDataset()
        print("Training the model...")
        model.trainModel(ds.getTrainingData())
        print("Evaluating the model...")
        model.evaluateModel(ds.getTestingData())
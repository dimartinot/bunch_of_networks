from Keras_Impl.Networks.MLP import MLPModel as MLP
from Keras_Impl.Networks.ConvNetImage import ConvNetImageModel as CV

from Keras_Impl.Datasets import cifar10Dataset, dummyDataset, mnistDataset

from termcolor import colored
import os

def exec():
    print(colored('------- KERAS MENU -------','green'))
    print('Please select the model you would like to train and, maybe, test:')
    print('\n1°/ MLP')
    print('\n1°/ ConvNet for images input (LeNet5)')
    print('')
    try:
        choice = int(input("Selection > "))
        os.system("cls")
    except:
        os.system("cls")
        print("The choice has to be an integer... Default selection = 1")
        choice = 1

    if (choice == 1):
        print(colored('------- MLP -------','red'))
        print("Loading the Dataset...")
        ds = mnistDataset.MNISTDataset()
        ds.flattenData()
        print(colored("Creating the model...",'blue'))
        model = CV.ConvNetImageModel()
        model.createModel(ds.getInputShape(), ds.getNumClasses())
        # ds = cifar10Dataset.Cifar10Dataset()
        # ds.flattenData()
        print("Training the model...")
        model.trainModel(ds.getTrainingData())
        print("Evaluating the model...")
        model.evaluateModel(ds.getTestingData())

    if (choice == 2):
        print(colored('------- ConvNet (LeNet5) -------','red'))
        print("Loading the Dataset...")
        ds = mnistDataset.MNISTDataset()
        ds.flattenData()
        print(colored("Creating the model...",'blue'))
        model = MLP.MLPModel()
        model.createModel(ds.getInputShape(), ds.getNumClasses())
        # ds = cifar10Dataset.Cifar10Dataset()
        # ds.flattenData()
        print("Training the model...")
        model.trainModel(ds.getTrainingData())
        print("Evaluating the model...")
        model.evaluateModel(ds.getTestingData())
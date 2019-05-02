from Keras_Impl.Networks.MLP import MLPModel as MLP
from Keras_Impl.Networks.ConvNetImage import convNetImageModel as CV
from Keras_Impl.Networks.AlexNet import alexNetModel as AL

from Keras_Impl.Datasets import cifar10Dataset, dummyDataset, mnistDataset

from termcolor import colored
import os

def exec():
    print(colored('------- KERAS MENU -------','green'))
    print('Please select the model you would like to train and, maybe, test:')
    print('\n1°/ MLP')
    print('\n2°/ ConvNet for images input (LeNet5)')
    print('\n3°/ ConvNet for images input (AlexNet)')
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

        MLP = MLP.MLPModel()
        MLP.createModel(ds.getInputShape(), ds.getNumClasses())
        
        MLP.model.summary()

        input("Launch training process...")
        print("Training the model...")

        MLP.trainModel(ds.getTrainingData())
        print("Evaluating the model...")

        MLP.evaluateModel(ds.getTestingData())

    if (choice == 2):
        print(colored('------- ConvNet (LeNet5) -------','red'))
        print("Loading the Dataset...")

        ds = mnistDataset.MNISTDataset()

        print(colored("Creating the model...",'blue'))

        LeNet5 = CV.ConvNetImageModel()
        LeNet5.createModel(ds.getInputShape(), ds.getNumClasses())

        LeNet5.model.summary()

        input("Launch training process...")
        print("Training the model...")

        model.trainModel(ds.getTrainingData())

        print("Evaluating the model...")

        model.evaluateModel(ds.getTestingData())

    if (choice == 3):
        print(colored('------- AlexNet -------','red'))
        print("Loading the Dataset...")

        ds = dummyDataset.DummyDataset((227,227,3),1000)

        print(colored("Creating the model...",'blue'))

        AlexNet = AL.AlexNetModel()
        AlexNet.createModel(ds.getInputShape(), ds.getNumClasses())
        
        AlexNet.model.summary()

        input("Launch training process...")
        print("Training the model...")

        AlexNet.trainModel(ds.getTrainingData())

        print("Evaluating the model...")

        AlexNet.evaluateModel(ds.getTestingData())
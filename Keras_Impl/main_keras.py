from Keras_Impl.Networks.MLP import MLPModel as MLP
from Keras_Impl.Networks.ConvNetImage import convNetImageModel as CV
from Keras_Impl.Networks.AlexNet import alexNetModel as AL
from Keras_Impl.Networks.VGG import vggModel as VGG

from Keras_Impl.Datasets import cifar10Dataset, dummyDataset, mnistDataset

from termcolor import colored
import os


def runModel(model, dataset=dummyDataset.DummyDataset((227,227,3), 1000), toFlatten=False):
    print(colored('------- {} -------'.format(model.getName()),'red'))
    print("Loading the Dataset...")

    if (toFlatten):
        dataset.flattenData()

    print(colored("Creating the model...",'blue'))

    model.createModel(dataset.getInputShape(), dataset.getNumClasses())
        
    rep = input("Would you like to check the model beforehand? (Y/N) > ")

    if rep == 'Y' or rep == 'y':
        model.model.summary()
    elif rep == 'N' or rep == 'n':
        print("Summary not printed..")
    else:
        print("Answer not understood: default behaviour chosen (rep=N)")

    input("Press any key to launch training process> ")
    print("Training the model...")

    model.trainModel(dataset.getTrainingData())
    print("Evaluating the model...")

    model.evaluateModel(dataset.getTestingData())

def exec():
    print(colored('------- KERAS MENU -------','green'))
    print('Please select the model you would like to train and, maybe, test:')
    print('\n1°/ MLP')
    print('\n2°/ ConvNet for image input (LeNet5)')
    print('\n3°/ ConvNet for image input (AlexNet)')
    print('\n4°/ ConvNet for image input (VGG16)')
    print('')
    try:
        choice = int(input("Selection > "))
        os.system("cls")
    except:
        os.system("cls")
        print("The choice has to be an integer... Default selection = 1")
        choice = 1

    if (choice == 1):
        runModel(model=MLP.MLPModel(), dataset=mnistDataset.MNISTDataset(), toFlatten=True)
        # print(colored('------- MLP -------','red'))
        # print("Loading the Dataset...")

        # ds = mnistDataset.MNISTDataset()
        # ds.flattenData()

        # print(colored("Creating the model...",'blue'))

        # MLP = MLP.MLPModel()
        # MLP.createModel(ds.getInputShape(), ds.getNumClasses())
        
        # MLP.model.summary()

        # input("Press any key to launch training process> ")
        # print("Training the model...")

        # MLP.trainModel(ds.getTrainingData())
        # print("Evaluating the model...")

        # MLP.evaluateModel(ds.getTestingData())

    if (choice == 2):
        runModel(model=CV.ConvNetImageModel(), dataset=mnistDataset.MNISTDataset(), toFlatten=False)
        # print(colored('------- ConvNet (LeNet5) -------','red'))
        # print("Loading the Dataset...")

        # ds = mnistDataset.MNISTDataset()

        # print(colored("Creating the model...",'blue'))

        # LeNet5 = CV.ConvNetImageModel()
        # LeNet5.createModel(ds.getInputShape(), ds.getNumClasses())

        # LeNet5.model.summary()

        # input("Press any key to launch training process> ")
        # print("Training the model...")

        # LeNet5.trainModel(ds.getTrainingData())

        # print("Evaluating the model...")

        # LeNet5.evaluateModel(ds.getTestingData())

    if (choice == 3):
        runModel(model=AL.AlexNetModel(), dataset=dummyDataset.DummyDataset((227,227,3),1000), toFlatten=False)

        # print(colored('------- AlexNet -------','red'))
        # print("Loading the Dataset...")

        # ds = dummyDataset.DummyDataset((227,227,3),1000)

        # print(colored("Creating the model...",'blue'))

        # AlexNet = AL.AlexNetModel()
        # AlexNet.createModel(ds.getInputShape(), ds.getNumClasses())
        
        # AlexNet.model.summary()

        # input("Press any key to launch training process> ")
        # print("Training the model...")

        # AlexNet.trainModel(ds.getTrainingData())

        # print("Evaluating the model...")

        # AlexNet.evaluateModel(ds.getTestingData())

    if (choice == 4):
        runModel(model=VGG.VGGModel(), dataset=dummyDataset.DummyDataset((224,224,3),1000), toFlatten=False)
        
        # print(colored('------- VGG -------','red'))
        # print("Loading the Dataset...")

        # ds = dummyDataset.DummyDataset((227,227,3),1000)



        # print(colored("Creating the model...",'blue'))

        # AlexNet = AL.AlexNetModel()
        # AlexNet.createModel(ds.getInputShape(), ds.getNumClasses())
        
        # AlexNet.model.summary()

        # input("Press any key to launch training process> ")
        # print("Training the model...")

        # AlexNet.trainModel(ds.getTrainingData())

        # print("Evaluating the model...")

        # AlexNet.evaluateModel(ds.getTestingData())
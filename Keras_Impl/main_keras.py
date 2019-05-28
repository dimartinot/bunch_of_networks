from Keras_Impl.Networks.MLP import MLPModel as MLP
from Keras_Impl.Networks.ConvNetImage import convNetImageModel as CV
from Keras_Impl.Networks.AlexNet import alexNetModel as AL
from Keras_Impl.Networks.VGG import vggModel as VGG
from Keras_Impl.Networks.GoogLeNet import googLeNetModel as GLN
from Keras_Impl.Networks.ResNet import resNetModel as RN

from Keras_Impl.Datasets import cifar10Dataset, dummyDataset, mnistDataset

from termcolor import colored
import os


def runModel(model, dataset=None, toFlatten=False):
    if (dataset is None):
        print("No dataset given, Dummy given by default")
        dataset = dummyDataset.DummyDataset((227,227,3), 1000) # We don't set it as the default value so that it is instantiated only if needed
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
    print('\n5°/ ConvNet for image input with Inception Modules (GoogLeNet)')
    print('\n6°/ Residual Network (ResNet 2015)')
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

    if (choice == 2):
        runModel(model=CV.ConvNetImageModel(), dataset=mnistDataset.MNISTDataset(), toFlatten=False)

    if (choice == 3):
        runModel(model=AL.AlexNetModel(), dataset=dummyDataset.DummyDataset((227,227,3),1000), toFlatten=False)

    if (choice == 4):
        runModel(model=VGG.VGGModel(), dataset=dummyDataset.DummyDataset((224,224,3),1000), toFlatten=False)

    if (choice == 5):
        runModel(model=GLN.GoogLeNetModel(), dataset=dummyDataset.DummyDataset((224,224,3),1000,num_of_outputs=3), toFlatten=False)

    if (choice == 6):
        runModel(model=RN.ResNetModel(), dataset=dummyDataset.DummyDataset((224,224,3),1000), toFlatten=False)
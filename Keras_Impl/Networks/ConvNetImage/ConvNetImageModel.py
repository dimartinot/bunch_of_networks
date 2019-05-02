import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from Keras_Impl.Networks import abstractModel as AM 

NUM_CLASSES = 10

NUM_OF_EPOCHS = 10

BATCH_SIZE = 32

DEFAULT_INPUT_SHAPE = (28, 28, 1)

class ConvNetImage(AM.Model):
    """
        This class implements a classic ConvNets model using Keras as the framework. It takes its architecture from LeCun's LeNet5
    """
    def __init__(self):
        super().__init__(self)
        self.model = KM.Sequential()

    def createModel(self, input_shape = None, num_classes = None):

        # Knowing this is not a good practice, we give the possibility to the user to change the default input shape from 1024 to any they want
        if (input_shape != None):
            default_input_shape = input_shape
        else:
            default_input_shape = DEFAULT_INPUT_SHAPE

        if (num_classes != None):
            default_num_classes = num_classes
        else:
            default_num_classes = DEFAULT_INPUT_SHAPE

        # Model type definition
        conv1 = KL.Conv2D(6, kernel_size=(5, 5), input_shape=(default_input_shape,), padding='valid', activation='relu')
        pool2 = KL.Pooling(pool_size=2)
        conv3 = KL.Conv2D(6, kernel_size=(5, 5), stride=(2,2), padding="valid", activation = 'relu')
        pool4 = KL.Pooling(pool_size = 2)
        flat5 = KL.Flatten()
        fc6 = KL.Dense(120, activation='relu')
        fc7 = KL.Dense(84, activation='relu')
        fc8 = KL.Dense(num_classes, activation='softmax')
        # Last layer has to have a softmax activation function

        self.model.add(conv1)
        self.model.add(pool2)
        self.model.add(conv3)
        self.model.add(pool4)
        self.model.add(flat5)
        self.model.add(fc6)
        self.model.add(fc7)
        self.model.add(fc8)

        # Compiling of the model
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def trainModel(self,train_dataset):
        """
            For self to work, data must be in the form :
            train_dataset = (x_train, y_train)
            test_dataset = (x_test, y_test)
        """
        (x_train, y_train) = train_dataset

        self.model.fit(x_train, y_train, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE)

    def evaluateModel(self,test_dataset):
        """
            For self to work, data must be in the form :
            test_dataset = (x_test, y_test)
        """
        (x_test, y_test) = test_dataset

        score = self.model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

        print(score)
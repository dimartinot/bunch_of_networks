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

DEFAULT_INPUT_SHAPE = (227, 227, 3)

VALIDATION_DATA = 0.2

class AlexNetModel(AM.Model):
    """
        This class implements a classic AlexNet model using Keras as the framework. It takes its architecture from Ilya Sutskever and Krizhevsky's network.
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
            default_num_classes = NUM_CLASSES

        conv1 = KL.Conv2D(96, kernel_size=(11, 11), input_shape=default_input_shape, strides=4, padding='valid', activation='relu')
        pool1 = KL.MaxPooling2D(pool_size=(3, 3), strides=2)

        conv2 = KL.Conv2D(256, kernel_size=(5, 5), padding='valid', activation="relu")
        pool2 = KL.MaxPooling2D(pool_size=(3, 3), strides=2)

        conv3 = KL.Conv2D(384, kernel_size=(3, 3), padding='valid', activation="relu")

        conv4 = KL.Conv2D(384, kernel_size=(3, 3), padding='valid', activation="relu")

        conv5 = KL.Conv2D(256, kernel_size=(3, 3), padding='valid', activation="relu")
        pool5 = KL.MaxPooling2D(pool_size=(3, 3), strides=2)

        flat6 = KL.Flatten()

        fc7 = KL.Conv2D(4096, kernel_size=(1, 1), padding='valid', activation="relu")
        fc8 = KL.Conv2D(4096, kernel_size=(1, 1), padding='valid', activation="relu")

        fc9 = KL.Dense(num_classes, activation="softmax")

        list_of_layers = [conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, flat6, fc7, fc8, fc9]

        for layer in list_of_layers:
            self.model.add(layer)



        self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    def trainModel(self, train_dataset):
        """
            For trainModel to work, data must be in the form :
            train_dataset = (x_train, y_train)
        """
        (x_tmp, y_tmp) = train_dataset

        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs/Keras_Impl/AlexNet', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)
        ]

        size_of_validation_data = int(x_tmp.shape[0]*VALIDATION_DATA)

        (x_validation, y_validation) = (x_tmp[:size_of_validation_data], y_tmp[:size_of_validation_data])
        (x_train, y_train) = (x_tmp[size_of_validation_data:], y_tmp[size_of_validation_data:])

        self.model.fit(x_train, y_train, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, validation_data=(x_validation, y_validation))

    def evaluateModel(self,test_dataset):
        """
            For self to work, data must be in the form :
            test_dataset = (x_test, y_test)
        """
        (x_test, y_test) = test_dataset

        score = self.model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

        print(score)

    def getName(self):
        return "AlexNet"
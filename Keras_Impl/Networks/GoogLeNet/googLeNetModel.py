import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import keras.optimizers as KO
from Keras_Impl.Networks import abstractModel as AM 
from Keras_Impl.Networks.GoogLeNet.Layers import LRN2D as CustomLayers

NUM_CLASSES = 10

NUM_OF_EPOCHS = 10

BATCH_SIZE = 32

DEFAULT_INPUT_SHAPE = (224, 224, 3)

VALIDATION_DATA = 0.2

class GoogLeNetModel(AM.Model):

    """
        This class implements the GoogLeNet model and its famous inception modules using Keras as the framework.
    """
    def __init__(self):
        super().__init__(self)
        self.model = None

    def createModel(self, input_shape = None, num_classes = None):

        def inceptionModule(number_of_filters, input_layer):
            """
                This function returns an inceptionModule as a list of Keras layers
            """

            # We create the first layer of convolution: 
            conv_1by1_1 = KL.Conv2D(number_of_filters['1_by_1'], (1,1), padding='same', activation='relu')(input_layer)

            conv_1by1_2 = KL.Conv2D(number_of_filters['3_by_3_reduce'], (1,1), padding='same', activation='relu')(input_layer)

            conv_1by1_3 = KL.Conv2D(number_of_filters['5_by_5_reduce'], (1,1), padding='same', activation='relu')(input_layer)

            maxPool2D = KL.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input_layer)

            # We then create the second layer of convolution : please note that the conv_1by1_1 will directly be concatenated with the other 3x3 and 5x5 convolutions
            conv_3by3 = KL.Conv2D(number_of_filters['3_by_3'], (1,1), padding='same', activation='relu')(conv_1by1_2)

            conv_5by5 = KL.Conv2D(number_of_filters['5_by_5'], (1,1), padding='same', activation='relu')(conv_1by1_3)

            conv_1by1_4 = KL.Conv2D(number_of_filters['after_pool'], (1,1), padding='same', activation='relu')(maxPool2D)

            # We then concatenate the output
            output = KL.concatenate([conv_1by1_1, conv_3by3, conv_5by5, conv_1by1_4], axis = 3)

            return output

        # Knowing this is not a good practice, we give the possibility to the user to change the default input shape from 1024 to any they want
        if (input_shape != None):
            default_input_shape = input_shape
        else:
            default_input_shape = DEFAULT_INPUT_SHAPE

        if (num_classes != None):
            default_num_classes = num_classes
        else:
            default_num_classes = NUM_CLASSES
        
        input_layer = KL.Input(shape = default_input_shape)

        conv1 = KL.Conv2D(64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(input_layer)

        pool2 = KL.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(conv1)

        batch3 = CustomLayers.LRN2D(batch_input_shape=(56,56,64))(pool2)

        conv4 = KL.Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(batch3)

        conv5 = KL.Conv2D(192, kernel_size=(3,3), padding='same', strides=1, activation='relu')(conv4)

        batch6 = CustomLayers.LRN2D(batch_input_shape=(56,56,192))(conv5)

        pool7 = KL.MaxPooling2D(pool_size=(3,3), padding='same', strides=2)(batch6)

        # First inception module, writing the number_of_filters:
        number_of_filters_inception_1 = {
            '1_by_1': 64,
            '3_by_3_reduce': 96, # the 1 by 1 conv before 3 by 3 layers
            '3_by_3': 128,
            '5_by_5_reduce': 16, # the 1 by 1 conv before 5 by 5 layers
            '5_by_5': 32,
            'after_pool':32
        }

        inception1 = inceptionModule(number_of_filters=number_of_filters_inception_1, input_layer=pool7)

        # Second inception module
        number_of_filters_inception_2 = {
            '1_by_1': 128,
            '3_by_3_reduce': 128,
            '3_by_3': 192,
            '5_by_5_reduce': 32,
            '5_by_5': 96,
            'after_pool':64
        }

        inception2 = inceptionModule(number_of_filters=number_of_filters_inception_2, input_layer=inception1)

        pool8 = KL.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(inception2)

        # Third inception module
        number_of_filters_inception_3 = {
            '1_by_1': 192,
            '3_by_3_reduce': 96,
            '3_by_3': 208,
            '5_by_5_reduce': 16,
            '5_by_5': 48,
            'after_pool':64
        }

        inception3 = inceptionModule(number_of_filters=number_of_filters_inception_3, input_layer=pool8)

        # Fourth inception module
        number_of_filters_inception_4 = {
            '1_by_1': 160,
            '3_by_3_reduce': 112,
            '3_by_3': 224,
            '5_by_5_reduce': 24,
            '5_by_5': 64,
            'after_pool':64
        }

        inception4 = inceptionModule(number_of_filters=number_of_filters_inception_4, input_layer=inception3)

        # Fifth inception module
        number_of_filters_inception_5 = {
            '1_by_1': 128,
            '3_by_3_reduce': 128,
            '3_by_3': 256,
            '5_by_5_reduce': 24,
            '5_by_5': 64,
            'after_pool':64
        }

        inception5 = inceptionModule(number_of_filters=number_of_filters_inception_5, input_layer=inception4)

        # Sixth inception module
        number_of_filters_inception_6 = {
            '1_by_1': 112,
            '3_by_3_reduce': 144,
            '3_by_3': 288,
            '5_by_5_reduce': 32,
            '5_by_5': 64,
            'after_pool':64
        }

        inception6 = inceptionModule(number_of_filters=number_of_filters_inception_6, input_layer=inception5)

        # Seventh inception module
        number_of_filters_inception_7 = {
            '1_by_1': 256,
            '3_by_3_reduce': 160,
            '3_by_3': 320,
            '5_by_5_reduce': 32,
            '5_by_5': 128,
            'after_pool': 128
        }

        inception7 = inceptionModule(number_of_filters=number_of_filters_inception_7, input_layer=inception6)

        pool9 = KL.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(inception7)

        # Eighth inception module
        number_of_filters_inception_8 = {
            '1_by_1': 256,
            '3_by_3_reduce': 160,
            '3_by_3': 320,
            '5_by_5_reduce': 32,
            '5_by_5': 128,
            'after_pool': 128
        }

        inception8 = inceptionModule(number_of_filters=number_of_filters_inception_8, input_layer=pool9)

        # Nineth inception module
        number_of_filters_inception_9 = {
            '1_by_1': 384,
            '3_by_3_reduce': 192,
            '3_by_3': 384,
            '5_by_5_reduce': 48,
            '5_by_5': 128,
            'after_pool': 128
        }

        inception9 = inceptionModule(number_of_filters=number_of_filters_inception_9, input_layer=inception8)

        pool10 = KL.AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(inception9)

        drop11 = KL.Dropout(rate=0.4)(pool10)

        fc12 = KL.Dense(num_classes, activation='softmax')(drop11)

        self.model = KM.Model(inputs=[input_layer], outputs=[fc12])

        # layers = [conv1, pool2, batch3, conv4, conv5, batch6, pool7, inception1, inception2, pool8, inception3, inception4, inception5, inception6, inception7, pool9, inception8, inception9, pool10, drop11, fc12]

        # for layer in layers:
        #     self.model.add(layer)

        optim = KO.SGD(lr=0.01, momentum=0.9, decay=0.04)

        self.model.compile(optimizer=optim, loss='categorical_crossentropy',metrics=['accuracy'])

    def trainModel(self, train_dataset):
        """
            For trainModel to work, data must be in the form :
            train_dataset = (x_train, y_train)
        """
        (x_tmp, y_tmp) = train_dataset

        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs/Keras_Impl/GoogLeNet', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)
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
        return "GoogLeNet"
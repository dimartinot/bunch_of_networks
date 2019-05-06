from Keras_Impl.Networks import abstractModel as AM 
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

NUM_CLASSES = 10

NUM_OF_EPOCHS = 10

BATCH_SIZE = 32

DEFAULT_INPUT_SHAPE = (224, 224, 3)

VALIDATION_DATA = 0.2

class VGGModel(AM.Model):
    """
        This class implements a classic VGG model using Keras as the framework. It takes its architecture from Oxford's renowned Visual Geometry Group
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

        conv1 = KL.Conv2D(64, input_shape=default_input_shape, kernel_size=(3, 3), padding='valid', activation='relu')
        conv2 = KL.Conv2D(64, kernel_size=(3, 3), activation='relu')

        pool3 = KL.MaxPooling2D(pool_size=(2, 2))

        conv4 = KL.Conv2D(128, kernel_size=(3, 3), activation='relu')
        conv5 = KL.Conv2D(128, kernel_size=(3, 3), activation='relu')

        pool6 = KL.MaxPooling2D(pool_size=(2, 2))

        conv7 = KL.Conv2D(256, kernel_size=(3, 3), activation='relu')
        conv8 = KL.Conv2D(256, kernel_size=(3, 3), activation='relu')
        conv9 = KL.Conv2D(256, kernel_size=(3, 3), activation='relu')

        pool10 = KL.MaxPooling2D(pool_size=(2, 2))

        conv11 = KL.Conv2D(512, kernel_size=(3, 3), activation='relu')
        conv12 = KL.Conv2D(512, kernel_size=(3, 3), activation='relu')
        conv13 = KL.Conv2D(512, kernel_size=(3, 3), activation='relu')

        pool14 = KL.MaxPooling2D(pool_size=(2, 2))

        conv15 = KL.Conv2D(512, kernel_size=(3, 3), activation='relu')
        conv16 = KL.Conv2D(512, kernel_size=(3, 3), activation='relu')
        conv17 = KL.Conv2D(512, kernel_size=(3, 3), activation='relu')       

        pool18 = KL.MaxPooling2D(pool_size=(2, 2))

        flat19 = KL.Flatten()

        fc20 = KL.Dense(4096, activation='relu')
        fc21 = KL.Dense(4096, activation='relu')

        out22 = KL.Dense(num_classes, activation='softmax')

        layers = [conv1, conv2, pool3, conv4, conv5, pool6, conv7, conv8, conv9, pool10, conv11, conv12, conv13, pool14, conv15, conv16, conv17, pool18, flat19, fc20, fc21, out22]
        
        for layer in layers:
            self.model.add(layer)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def trainModel(self,train_dataset):

        (x_tmp, y_tmp) = train_dataset

        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs/Keras_Impl/VGG', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)
        ]

        size_of_validation_data = int(x_tmp.shape[0]*VALIDATION_DATA)

        (x_validation, y_validation) = (x_tmp[:size_of_validation_data], y_tmp[:size_of_validation_data])
        (x_train, y_train) = (x_tmp[size_of_validation_data:], y_tmp[size_of_validation_data:])

        self.model.fit(x_train, y_train, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, validation_data=(x_validation, y_validation))

    
    def evaluateModel(self,test_dataset):

        (x_test, y_test) = test_dataset

        score = self.model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

        print(score)

    def getName(self):
        return "VGG16"
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import keras.optimizers as KO
from Keras_Impl.Networks import abstractModel as AM 

NUM_CLASSES = 10

NUM_OF_EPOCHS = 10

BATCH_SIZE = 32

DEFAULT_INPUT_SHAPE = (152, 152, 3)

VALIDATION_DATA = 0.2

class DeepFaceModel(AM.Model):

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

        input_layer = KL.Input(shape = default_input_shape)

        #padding1 = KL.ZeroPadding3D(padding=(0,0,1))(input_layer) # this is used to transform the 152*152*3 input into 152*152*5 so that the 3D Convolution doesn't chunk the tensor to 142*142*1

        conv1 = KL.Conv2D(32, (11, 11), activation='relu',padding='valid')(input_layer)

        padding2 = KL.ZeroPadding2D(padding=(1,1))(conv1) # this is used so that the result of the MaxPooling is in the due 71*71 format

        pool2 = KL.MaxPooling2D(pool_size=3, strides=2)(padding2)

        conv3 = KL.Conv2D(16, (9, 9), activation='relu')(pool2)

        local4 = KL.LocallyConnected2D(16, (9, 9), activation='relu')(conv3)

        local5 = KL.LocallyConnected2D(16, (7, 7), strides=2, activation='relu')(local4) # without the strides, the dimension is not the same as specified in the paper, but there is no trace of such a stride in the said paper..

        local6 = KL.LocallyConnected2D(16, (5, 5), activation='relu')(local5)

        flat7 = KL.Flatten()(local6)

        fc8 = KL.Dense(4096, activation='relu')(flat7)

        fc9 = KL.Dense(4030, activation='relu')(fc8)

        output = KL.Dense(default_num_classes, activation='softmax')(fc9)

        self.model = KM.Model(inputs=[input_layer], outputs=[output])

        sgd = KO.SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)

        # Compiling the model
        self.model.compile(optimizer=sgd,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def trainModel(self, train_dataset):
        """
            For trainModel to work, data must be in the form :
            train_dataset = (x_train, y_train)
        """
        (x_tmp, y_tmp) = train_dataset

        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs/Keras_Impl/DeepFace', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)
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
        return "Locally Connected Network (DeepFace)"
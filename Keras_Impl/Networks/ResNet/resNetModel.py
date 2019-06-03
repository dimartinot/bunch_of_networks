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

DEFAULT_INPUT_SHAPE = (224, 224, 3)

VALIDATION_DATA = 0.2

class ResNetModel(AM.Model):

    """
        This class implements the 2015 ResNet using keras framework.
        For its specificities, I have mainly followed the advices of their 2015 paper.
        About the residual connections, they explained:
            "The identity shortcuts (Eqn.(1)) can be directly used when the input andoutput are of the 
            same dimensions (solid line shortcuts inFig. 3). When the dimensions increase (dotted line 
            shortcutsin Fig. 3), we consider two options:  (A) The shortcut stillperforms identity mapping, 
            with extra zero entries paddedfor increasing dimensions.  This option introduces no extraparameter; 
            (B) The projection shortcut in Eqn.(2) is used tomatch dimensions (done by 1×1 convolutions)."

        For my implementation I have chosen to use 1x1 convolutions to match the required size
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

        input_layer = KL.Input(shape = default_input_shape)

        conv1 = KL.Conv2D(64, (7, 7), strides=2, padding='same')(input_layer)

        pool2 = KL.MaxPooling2D((2, 2), strides=2)(conv1)

        conv3 = KL.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)

        conv4 = KL.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

        #First residual layer
        res1 = KL.add([pool2, conv4])

        conv5 = KL.Conv2D(64, (3, 3), activation='relu', padding='same')(res1)

        conv6 = KL.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

        #Second residual layer
        res2 = KL.add([res1, conv6])

        conv7 = KL.Conv2D(64, (3, 3), activation='relu', padding='same')(res2)

        conv8 = KL.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

        # Switching the number of filters + increasing dimensions
        #Third residual layer

        res3 = KL.add([conv8, res2])#(54, 54, 64) (27, 27, 128)

        conv9 = KL.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(res3)

        conv10 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)

        #Fourth residual layer
        #We project the former residual layer to match the dimension of the new convolutions (dotted lines in the paper)
        projection_shortcut1 = KL.Conv2D(128, (1, 1), strides=2, padding='same')(res3)
        res4 = KL.add([projection_shortcut1, conv10])

        conv11 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(res4)

        conv12 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(conv11)

        #Fifth residual layer
        res5 = KL.add([res4, conv12])

        conv13 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(res5)

        conv14 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(conv13)

        #6th residual layer
        res6 = KL.add([res5, conv14])

        conv15 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(res6)

        conv16 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(conv15)

        #7th residual layer
        res7 = KL.add([res6, conv16])

        conv17 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(res7)

        conv18 = KL.Conv2D(128, (3, 3), activation='relu', padding='same')(conv17)

        # Switching the number of filters + increasing dimensions
        #8th residual layer
        res8 = KL.add([res7, conv18])

        conv19 = KL.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(res8)

        conv20 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(conv19)

        #9th residual layer
        #We project the former residual layer to match the dimension of the new convolutions (dotted lines in the paper)
        projection_shortcut2 = KL.Conv2D(256, (1, 1), strides=2, padding='same')(res8)
        res9 = KL.add([projection_shortcut2, conv20])

        conv21 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(res9)

        conv22 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(conv21)

        #10th residual layer
        res10 = KL.add([res9, conv22])

        conv23 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(res10)

        conv24 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(conv23)

        #11th residual layer
        res11 = KL.add([res10, conv24])

        conv25 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(res11)

        conv26 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(conv25)

        #12th residual layer
        res12 = KL.add([res11, conv26])

        conv27 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(res12)

        conv28 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(conv27)

        #13th residual layer
        res13 = KL.add([res12, conv28])

        conv29 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(res13)

        conv30 = KL.Conv2D(256, (3, 3), activation='relu', padding='same')(conv29)

        # Switching the number of filters (512) + increasing dimensions
        #14th residual layer
        res14 = KL.add([res13, conv30])

        conv31 = KL.Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(res14)

        conv32 = KL.Conv2D(512, (3, 3), activation='relu', padding='same')(conv31)

        #15th residual layer
        #We project the former residual layer to match the dimension of the new convolutions (dotted lines in the paper)
        projection_shortcut3 = KL.Conv2D(512, (1, 1), strides=2, padding='same')(res14)
        res15 = KL.add([projection_shortcut3, conv32])

        conv33 = KL.Conv2D(512, (3, 3), activation='relu', padding='same')(res15)

        conv34 = KL.Conv2D(512, (3, 3), activation='relu', padding='same')(conv33)

        #16th residual layer
        res16 = KL.add([res15, conv34])

        conv35 = KL.Conv2D(512, (3, 3), activation='relu', padding='same')(res16)

        conv36 = KL.Conv2D(512, (3, 3), activation='relu', padding='same')(conv35)

        # Global Average pooling : size of the filters based on the size of the output of last convolution. It has the effect of a flatten
        pool37 = KL.AveragePooling2D(pool_size=(7, 7))(conv36)

        flat38 = KL.Flatten()(pool37) # to make sure that the input tensor of fc38 is of size (?, 1000) and not (?, 1, 1, 1000)

        fc39 = KL.Dense(default_num_classes, activation='softmax')(flat38)

        self.model = KM.Model(inputs=[input_layer], outputs=[fc39])

        sgd = KO.SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)

        # Compiling of the model
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
            keras.callbacks.TensorBoard(log_dir='./logs/Keras_Impl/ResNet', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)
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
        return "Residual Network (Original ResNet)"
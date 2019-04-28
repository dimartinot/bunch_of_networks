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

class MLPModel(AM.Model):

    def __init__(self):
        super().__init__(self)
        self.model = KM.Sequential()

    def createModel(self):
        # Model type definition
        fc1 = KL.Dense(32, input_shape=(1024,), activation='relu')
        dp2 = KL.Dropout(0.25)
        fc3 = KL.Dense(16, activation='relu')
        dp4 = KL.Dropout(0.5)
        # Last layer has to have a softmax activation function
        fc5 = KL.Dense(NUM_CLASSES,activation='softmax')

        self.model.add(fc1)
        self.model.add(dp2)
        self.model.add(fc3)
        self.model.add(dp4)
        self.model.add(fc5)

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
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras import Model

class myModel(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)




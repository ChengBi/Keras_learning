import keras
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np

mnist_train = np.load('../../data/mnist-train.npz')
mnist_valid = np.load('../../data/mnist-valid.npz')

class lenet():
    
    def __init__(self, train_data, valid_train):
        
        self.train_inputs = train_data['inputs'].reshape(-1, 28, 28, 1)
        self.valid_inputs = valid_train['inputs'].reshape(-1, 28, 28, 1)
        self.train_targets = keras.utils.to_categorical(train_data['targets'], num_classes=10)
        self.valid_targets = keras.utils.to_categorical(valid_train['targets'], num_classes=10)
        self.model = Sequential()
        self.model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), input_shape=(28, 28, 1)))  #14x14
        self.model.add(keras.layers.convolutional.Convolution2D(64, 5, 5))   #10x10
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2)))  #5x5
        self.model.add(keras.layers.convolutional.Convolution2D(256, 5, 5))   #1x1
        self.model.add(keras.layers.Flatten()) 
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(256))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(64))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(10))
        self.model.add(keras.layers.Activation('softmax'))
                       
        
    def run(self, epochs):
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.model.fit(self.train_inputs, self.train_targets, epochs = epochs, batch_size = 64)
    
    def evaluate(self):    
        score = self.model.evaluate(self.valid_inputs, self.valid_targets, batch_size = 64)
        print('------------------------------------')
        print(score)
        print('------------------------------------')


if __name__ == '__main__':
    
    bp = lenet(mnist_train, mnist_valid)
    for i in range(10):
        bp.run(5)
        bp.evaluate()
    
 
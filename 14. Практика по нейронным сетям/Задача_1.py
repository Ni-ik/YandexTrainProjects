from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, AvgPool2D, Conv2D, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam

def load_train(path):
    datagen = ImageDataGenerator(validation_split=0.25,
                                         horizontal_flip=True,###
                                         vertical_flip=True,###
                                         rescale = 1./255) 
    train_datagen_flow = datagen.flow_from_directory(path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345)
    #print('load')
    return train_datagen_flow



def create_model(input_shape):
    optimizer = Adam(lr=0.0001)
    model = Sequential()
    model=Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=optimizer, metrics=['acc'])
    return model




def train_model(model, train_data, test_data, batch_size=16, epochs=1,
               steps_per_epoch=None, validation_steps=None):

    model.fit(train_data, # < напишите код здесь >
              validation_data=test_data,# < напишите код здесь >,
              # Чтобы обучение не было слишком долгим, указываем
              # количество шагов равным 1
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model 


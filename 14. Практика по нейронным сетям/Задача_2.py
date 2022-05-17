from tensorflow.keras.layers import Dense, AvgPool2D, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50

def load_train(path):
    datagen = ImageDataGenerator(#validation_split=0.25,
                                         horizontal_flip=True,###
                                         vertical_flip=True,###
                                         #rotation_range = 90,
                                         rescale = 1./255) 
    train_datagen_flow = datagen.flow_from_directory(path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',                      
        #subset='training',
        seed=12345)
    #print('load')
    return train_datagen_flow


def create_model(input_shape):
   
    
    backbone = ResNet50(input_shape=input_shape,
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', #такие веса
                    include_top=False,
                       classes = 1000)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.7))
    model.add(Dense(12, activation='softmax')) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

  
def train_model(model, train_data, test_data, batch_size=None, epochs=10,
               steps_per_epoch=None, validation_steps=None):
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
        
    model.fit(train_data, 
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model 


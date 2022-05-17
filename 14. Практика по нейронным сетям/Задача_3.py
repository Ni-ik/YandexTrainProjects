from tensorflow.keras.layers import Dense, AvgPool2D, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50



def load_train(path):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    #labels = pd.read_csv(path + 'labels.csv')
    labels = pd.read_csv('/datasets/faces/labels.csv')
    _directory='/datasets/faces/final_files/'
    train_datagen_flow = datagen.flow_from_dataframe(dataframe=labels,
                                                     #directory=path + '/final_files/', 
                                                     directory = _directory,
                                                     x_col='file_name',
                                                     y_col='real_age',
                                                     target_size=(224, 224), 
                                                     batch_size=32, 
                                                     class_mode='raw', 
                                                     subset='training',
                                                     seed=12345)
    return train_datagen_flow


# In[12]:


def load_test(path):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    #labels = pd.read_csv(path + 'labels.csv')
    labels = pd.read_csv('/datasets/faces/labels.csv')
    test_datagen_flow = datagen.flow_from_dataframe(dataframe=labels,
                                                     #directory=path + '/final_files/', 
                                                     directory = _directory,
                                                     x_col='file_name',
                                                     y_col='real_age', 
                                                     target_size=(224, 224), 
                                                     batch_size=32, 
                                                     class_mode='raw', 
                                                     subset='validation',
                                                     seed=12345)
    return test_datagen_flow


# In[3]:


def create_model(input_shape):
   
    
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
   # model.add(Dropout(0.7))
    model.add(Dense(1, activation='relu'))
    #optimizer = Adam(lr=0.0001) 
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model


# In[3]:


def train_model(model, train_data, test_data, batch_size=None, epochs=5,
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
              verbose=2, shuffle=True)
    return model 


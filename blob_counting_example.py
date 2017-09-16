import cv2
import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.layers.core import Reshape
from keras import layers
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

import tensorflow as tf

import sys
import os
import shutil

#Single head predicting number of blobs on image

#params
max_number_of_objects= 12 #Maximum number objects on image
max_object_radius= 15
batch_size= 32
epochs= 150
patience = 10
model_name= 'model.h5'
image_h= 100
image_w= 120
input_dim= (image_h,image_w,3)

def generate_data():
    img= np.zeros(input_dim, np.uint8)
        
    #Draw non intersected circles
    number_of_objects=0
    for i in range(max_number_of_objects):
        rand_y= np.random.randint(image_h) 
        rand_x= np.random.randint(image_w)
        rand_radius= max(np.random.randint(max_object_radius),3)
        if(rand_x-rand_radius > 0 and rand_y-rand_radius > 0 and rand_x+rand_radius < image_w and rand_y+rand_radius < image_h):
            temp= np.zeros(input_dim, np.uint8)
            cv2.circle(temp, (rand_x,rand_y), rand_radius, (255,255,255), -1)
        
            if(np.sum(cv2.bitwise_and(temp,img)) == 0):
                img= cv2.bitwise_or(temp,img)
                
                number_of_objects=number_of_objects+1
    
    #print('number_of_circles', number_of_circles) #
    
    y= number_of_objects
    
    return img,y
    
def batch_generator(batch_size):
    while True:
        image_list = []
        y_list = []
        for i in range(batch_size):
            img,y= generate_data()
            image_list.append(img)
            y_list.append(y)
            
        image_arr = np.array(image_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int32)
        
        yield(image_arr, {'head': y_arr} )
    
def get_model(): 

    network_input = Input(shape=input_dim)
    
    conv1= Conv2D(32, (3, 3), padding='same')(network_input)
    pool1= MaxPooling2D(pool_size=(2, 2))(conv1)
    act1= Activation('relu')(pool1)
    
    conv2= Conv2D(32, (3, 3), padding='same')(act1)
    pool2= MaxPooling2D(pool_size=(2, 2))(conv2)
    act2= Activation('relu')(pool2)
    
    conv3= Conv2D(64, (3, 3), padding='same')(act2)
    pool3= MaxPooling2D(pool_size=(2, 2))(conv3)
    act3= Activation('relu')(pool3)
    
    conv4= Conv2D(64, (3, 3), padding='same')(act3)
    pool4= MaxPooling2D(pool_size=(2, 2))(conv4)
    act4= Activation('relu')(pool4)
    
    tail= Flatten()(act4)
    
    head= Dense(1, name='head')(tail)

    model = Model(input = network_input, output = [head])
    model.compile(optimizer = Adadelta(), loss = 'mse')
    
    #print(model.summary()) #
    #plot_model(model, to_file='model.png', show_shapes=True) #
    #sys.exit() #
    
    return model
    
def train():
    
    model= get_model()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0)
    ]
    
    history = model.fit_generator(
        generator=batch_generator(batch_size),
        nb_epoch=epochs,
        samples_per_epoch=1*batch_size,
        validation_data=batch_generator(batch_size),
        nb_val_samples=1*batch_size,
        verbose=1,
        callbacks=callbacks)

    model.save_weights(model_name)
    pd.DataFrame(history.history).to_csv('train_history.csv', index=False)
      
def evaluate_model():
    
    if(os.path.exists('generated_samples')):
        shutil.rmtree('generated_samples')
    os.makedirs('generated_samples')

    model= get_model()
    model.load_weights(model_name)
        
    n_samples=3
    for i in range(0,n_samples):
        img,y= generate_data()
        y_pred= model.predict(img[None,...])[0]
              
        print('n_objects ', y)
        print('n_objects estimated', int(y_pred[0]))
      
        cv2.imwrite(os.path.join('generated_samples','sample_'+str(i)+'.png'),img)
    
###########################################################################################
train()
evaluate_model()

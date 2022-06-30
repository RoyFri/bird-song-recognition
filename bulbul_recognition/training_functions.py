# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:22:16 2022

@author: royf2
"""
import numpy as np

# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import librosa.display as dsp

from matplotlib import pyplot as plt

import random



def prepare_data(data):
    '''
    gets processed data
    returns ds as a list: [X_train, X_test, y_train, y_test]
    '''
    train_labels = []
    for i in data:
        train_labels.append(i[2])
        
    train_data = []
    for i in data:
        train_data.append(i[3])

    # from lists to numpy arrays, 
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    # fitting data's shape to CNN's input layer
    data_shape = train_data.shape
    train_data = train_data.reshape((data_shape[0], data_shape[1], data_shape[2], 1))
    
    # -------- a try -----------
    # train_data = train_data[:10000]
    # train_labels = train_labels[:10000]
    
    ds = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    
    return ds



def train(X_train, X_test, y_train, y_test, batch_size=10, epochs=3, plot=True):
    '''
    gets ds as a list: [X_train, X_test, y_train, y_test]
    plots random examples from the data set (both 0 and 1 labeled)
    builds and trains a CNN model on the data
    plots loss and accuracy progress graph
    returns the trained model
    '''
   
    input_shape = X_train[0].shape
    train_len = len(X_train)
    
    # plot 5 random spectrograms of each label
    if plot:
        for i in range(10):
            if i < 5:
                while(1):
                    random_i = random.randrange(train_len)
                    if y_train[random_i] == 0:
                        break
            else:
                while(1):
                    random_i = random.randrange(train_len)
                    if y_train[random_i] == 1:
                        break
    
            log_mel_spectogram = X_train[random_i].reshape(input_shape[0], input_shape[1])
            dsp.specshow(log_mel_spectogram,
                         sr=44100,
                         x_axis='time',
                         y_axis='mel',
                         cmap='magma')
            title = "train example no. " + str(random_i) + " (label: " + str(y_train[random_i]) + ")"
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')
            plt.show()
    
    # build and train
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid',
               kernel_initializer=initializers.he_normal(),
               input_shape=input_shape),        
        MaxPool2D(pool_size=(2, 2), strides=2),
        Dropout(0.2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'),        
        MaxPool2D(pool_size=(2, 2), strides=2),
        Dropout(0.2),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'),    
        MaxPool2D(pool_size=(2, 2), strides=2),
        Dropout(0.2),
        Flatten(),
        Dense(units=32, activation='relu'),
        # Dropout(0.3),
        Dense(units=2, activation='softmax')])
        # Dense(units=1, activation = 'sigmoid')])

    model.summary()

    # model.compile(optimizer="rmsprop",
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model.compile(optimizer=Adam(learning_rate=0.001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    

    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()
    
    return model
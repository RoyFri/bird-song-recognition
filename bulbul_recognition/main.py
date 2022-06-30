# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 19:40:19 2022

@author: royf2
"""

import os
project_dir = r"C:\Users\royf2\.spyder-py3\project_bird_shazam\bulbul_recognition"
os.chdir(project_dir)
import pickle

import numpy as np

from tensorflow import keras
from sklearn.metrics import confusion_matrix

import data_processing_Bulbul5_lite as dpb
import training_functions as tf

from time import strftime



def main():
    
    #%% creating data files
    
    # dur = 0.5
    processed_data = dpb.data_processing(
        data_path  = 'train',
        min_xtrain = -80.0, 
        fs         = 44100,
        dur        = 0.5, 
        n_fft      = 2048,
        hop        = 700,
        n_filters  = 50,
        f_min      = 1000,
        f_max      = 3500,
        fL         = 700,
        fH         = 3900 ,
        Th         = 0.35)

    # Pickling
    with open("data1.pickle", "wb") as fp:
        pickle.dump(processed_data, fp)
        
        
    # dur = 1
    processed_data = dpb.data_processing(
        data_path  = 'train',
        min_xtrain = -80.0, 
        fs         = 44100,
        dur        = 1, 
        n_fft      = 2048,
        hop        = 700,
        n_filters  = 50,
        f_min      = 1000,
        f_max      = 3500,
        fL         = 700,
        fH         = 3900 ,
        Th         = 0.35)

    # Pickling
    with open("data2.pickle", "wb") as fp:
        pickle.dump(processed_data, fp)
        

    # dur = 0.75
    processed_data = dpb.data_processing(
        data_path  = 'train',
        min_xtrain = -80.0, 
        fs         = 44100,
        dur        = 0.75, 
        n_fft      = 2048,
        hop        = 700,
        n_filters  = 50,
        f_min      = 1000,
        f_max      = 3500,
        fL         = 700,
        fH         = 3900 ,
        Th         = 0.35)

    # Pickling
    with open("data3.pickle", "wb") as fp:
        pickle.dump(processed_data, fp)  
        
    
    # dur = 1, 64 filter banks
    processed_data = dpb.data_processing(
        data_path  = 'train',
        min_xtrain = -80.0, 
        fs         = 44100,
        dur        = 1, 
        n_fft      = 2048,
        hop        = 700,
        n_filters  = 64,
        f_min      = 1000,
        f_max      = 3500,
        fL         = 700,
        fH         = 3900 ,
        Th         = 0.35)

    # Pickling
    with open("data4.pickle", "wb") as fp:
        pickle.dump(processed_data, fp)



    #%% extracting data files
    
    # Unpickling data1 (dur = 0.5)
    with open("data1.pickle", "rb") as fp:
        processed_data_retrieve = pickle.load(fp) 
    
    # Unpickling data2 (dur = 1)
    with open("data2.pickle", "rb") as fp:
        processed_data_retrieve = pickle.load(fp)
        
    # Unpickling data3 (dur = 0.75)
    with open("data3.pickle", "rb") as fp:
        processed_data_retrieve = pickle.load(fp)
        
    # Unpickling data4 (dur = 1, 64 FB)
    with open("data4.pickle", "rb") as fp:
        processed_data_retrieve = pickle.load(fp)
        
        
    
    #%% preparing dataset
    
    ds = tf.prepare_data(processed_data_retrieve)
    X_train, X_test, y_train, y_test = ds
    #%% train, test and evaluate

    model = tf.train(X_train, X_test, y_train, y_test, batch_size=64, epochs=1, plot=False)
    
    # save and load model
    # time = strftime("%d-%m-%Y_%H-%M-%S")
    # model.save('model_'+time)
    saved_model = keras.models.load_model('model_03-03-2022_13-48-55')
    
    # print("\nTest evaluation:")
    # test_loss, test_acc = saved_model.evaluate(X_test, y_test)
    
    predictions = saved_model.predict(X_test)
    y_true = y_test
    y_pred = np.argmax(predictions, axis=-1)
    compare = y_pred == y_true
    num_same = np.count_nonzero(compare == True)
    test_accuracy = num_same / len(y_test)
    
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    num_neg = np.count_nonzero(y_test == 0)
    num_pos = np.count_nonzero(y_test == 1)
    true_neg  = cm[0,0]
    true_pos  = cm[1,1]
    false_neg = cm[1,0]
    false_pos = cm[0,1]
    true_pos_rate  = true_pos / num_pos  # successful bulbul recognition
    true_neg_rate  = true_neg / num_neg  # successful non-bulbul recognition
    false_pos_rate = false_pos / num_neg # wrongly recognized as bulbul
    false_neg_rate = false_neg / num_pos # wrongly recognized as non-bulbul
    
    FP_TP_ratio    = false_pos / true_pos
      
    print(f"\nTest accuracy: {test_accuracy}")   
    print(f"\ntrue_pos_rate: {true_pos_rate}")
    print(f"true_neg_rate: {true_neg_rate}")
    print(f"false_pos_rate: {false_pos_rate}")
    print(f"false_neg_rate: {false_neg_rate}")
    print()
    print(f"FP/TP ratio: {FP_TP_ratio}")
    
    
#%%
if __name__ == "__main__":
    main()
    
    
    
    








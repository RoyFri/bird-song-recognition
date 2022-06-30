# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:51:45 2022

@author: royf2

based on a code file of AM and YL, called data_processing_Bulbul5.py
"""


import librosa
import librosa.display as dsp
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import parsing_functions as pf
from scipy import signal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras as keras
import glob
import audio_functions as af
import Utils as Ut



def data_processing(data_path = 'data/train', min_xtrain = -80.0, 
                    fs = 44100, dur = 0.5,  n_fft = 2048, hop = 700,
                    n_filters = 50, f_min = 1000,
                    f_max = 3500, fL = 700, fH = 3900 , Th = 0.35):
    '''
    based on a function written by AM and YL
    gets a path to a folder contains recording wav files and their corresponding label txt files.
    extracts labels from the txt files
    builds a stream of frames. from each frame - creats a mel-specrtogram and determines its label.
    frame after frame, builds the data set and returns it
    '''

    frame_length = int(dur*fs)
    hop_length = int(frame_length / 2)
    
    # get list of all file paths in data_path

    wav_file_paths = glob.glob(data_path + "/*.wav", recursive=True)
    txt_file_paths = glob.glob(data_path + "/*.txt", recursive=True)
    num_files = len(wav_file_paths)
    
    processed_data = []
    for file_i in range(num_files):
        print(f"processing file no. {file_i}")
        # list of: ['label starting time', 'label ending time']:
        labels_by_sec = pf.read_label_file(txt_file_paths[file_i])
        # list of tuples: (label starting sample, label ending sample):
        labels_by_sample = [(int(fs*float(label[0])), int(fs*float(label[1]))) for label in labels_by_sec]
                 
        # Stream the data, working on 1 frame at a time
        stream = librosa.stream(
            wav_file_paths[file_i],
            block_length = 1, # each block contains 1 frame
            frame_length = frame_length,
            hop_length   = hop_length,
            fill_value   = 0) # filling last block with silence if needed     
        
        frame_i = 0
        for y in stream:
            bin_label = af.get_bin_label(frame_i, frame_length, hop_length, labels_by_sample) 
            ##calculate mel spectogram
            M = librosa.feature.melspectrogram(
                y,
                sr         = fs,
                n_fft      = n_fft,
                hop_length = hop,
                n_mels     = n_filters,       
                fmax       = f_max,
                fmin       = f_min)          
            
            M_med = Ut.medclip(M, 3.5, 1e-14)
            M_dB = librosa.power_to_db(M_med, ref=np.max)
            M_dB_clean = Ut.blobRemove(M_dB, 2, floordB = -80, Thresh = -60 )
            
            processed_data.append([file_i , frame_i , bin_label , M_dB_clean])
            
            frame_i += 1
            
    return processed_data # large_df #, df_aug #, MFBmat, bin_labels, predictions

def add_noise(x):
    '''Add random noise to an image'''
    var = 5.0
    deviation = var*np.random.random()
    noise = np.random.normal(0, deviation, x.shape)
    y = x + noise
    y = np.clip(y, -80., 0.)
    return y

 
def data_augmentation(x_train, augment_factor = 0.35):
    data_aug = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
        
    layers.experimental.preprocessing.RandomZoom(height_factor=(-0.2, -0.01), 
            width_factor = (-0.2, -0.01), fill_mode='reflect', interpolation='bilinear' )])
    
    X_train_Aug = [] 
    for ind in range(x_train.shape[0]):
    #         randnum = np.random.rand()
    #         augmented_images = data_aug(x_train[ind])
    #         X_train_Aug.append(augmented_images)
        randnum = np.random.rand()
        if randnum > augment_factor:
            augmented_images = data_aug(x_train[ind])
            X_train_Aug.append(augmented_images)  
        else:
            augmented_images = add_noise(x_train[ind])
            X_train_Aug.append(augmented_images)
            
########----check Augmentation-----######
        # plt.subplot(211)
        # plt.imshow(augmented_images[0])
        # plt.subplot(212)
        # orig_spect = x_train[ind]
        # plt.imshow(orig_spect[0,:,:,:])
        # plt.pause(1) 
        
    return X_train_Aug



# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:19:47 2021

@author: AM and YL
""" 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

    
def f0_synth(sig, siglen,f0, fs, frame_length):
    """     
    Parameters: x - original signal, siglen - length of the audio signal. 
    f0 - np array of pitch
    contour. fs - sampling frequency. frame_length - length of each frame
    for sinusoidal signal synthesis.
    ----------
    returns - y - a numpy array signal which composed of concateneation of
    sinusoidal signals, one for each frame, with continous phase in the boundaries
    between two consecutive sinusoids.

    """
    # import numpy as np
    # from scipy.signal import hilbert
    
    j=0
    phi = 0
    y = np.zeros(siglen)
    Ts = 1 / fs
    tt = np.linspace(0, float(frame_length/fs), frame_length)
    N = tt.size
    for i in range(0, siglen, frame_length):
        if j >= f0.size:
            break
        freq = f0[j]
        T0 = 1 / freq
        framesin=np.cos(2*np.pi*freq*tt+phi)
        y[i:i+frame_length] = framesin
        # if j>2:
        #     plt.plot(y[i-2*frame_length:i+2*frame_length])
        #     plt.pause(0.1)
        if framesin[-1] < framesin[-2]:
            phi = np.arccos(framesin[-1])+ Ts/T0*2*np.pi
        else:
            phi = 2*np.pi-np.arccos(framesin[-1])+ Ts/T0*2*np.pi          
        j = j + 1
    #plt.pause(0.1)    
    analytic_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    # plt.plot(amplitude_envelope)
    # plt.show()
    y = y*amplitude_envelope[0:y.size]
    
    return y

def chirp_synth(sig, siglen,f0, fs, frame_length, coeffs):
    """     
    Parameters: x - original signal, siglen - length of the audio signal. 
    f0 - np array of pitch contour. fs - sampling frequency. 
    frame_length - length of each frame for polynomial chirp signal synthesis.
    coeffs - the Legendre polynomial coefficients
    ----------
    returns - y - a numpy array signal of a chirp signal

    """
    # import numpy as np
    # from scipy.signal import hilbert
    
    j=0
    phi = 0
    y = np.zeros(siglen)
    Ts = 1 / fs
    # tt = np.linspace(0, float(frame_length/fs), frame_length)
    tt = np.linspace(0, float(siglen/fs), siglen)
    N = tt.size
    y = np.cos(2*np.pi*(coeffs[0]*tt+coeffs[1]/2*tt**2+coeffs[2]/3*tt**3)+coeffs[3]/4*tt**4)
    plt.plot(tt[:1000],y[:1000])
    
    # for i in range(0, siglen, frame_length):
    #     if j >= f0.size:
    #         break
    #     freq = f0[j]
    #     T0 = 1 / freq
    #     framesin=np.cos(2*np.pi*freq*tt+phi)
    #     y[i:i+frame_length] = framesin
    #     if framesin[-1] < framesin[-2]:
    #         phi = np.arccos(framesin[-1])+ Ts/T0*2*np.pi
    #     else:
    #         phi = 2*np.pi-np.arccos(framesin[-1])+ Ts/T0*2*np.pi          
    #     j = j + 1
        
    analytic_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    # plt.plot(amplitude_envelope)
    # plt.show()
    y = y*amplitude_envelope[0:y.size]
    
    return y
def Energy(sig, fs, frame_length, hop_size):
    """
    
    """
    import numpy as np
    import sys, os
    import matplotlib.pyplot as plt
    import re 
    from scipy import signal
    from scipy.signal import hilbert
    
    siglen = sig.size
    print(siglen)
    N = int(np.ceil(siglen/hop_size))
    E=np.zeros(N)    
    j=0
    y = np.zeros(int(siglen))    
    for i in range(0, siglen, hop_size):
        if i>=siglen:
            break
        E[j] = 1/frame_length*np.sum(sig[i:i+frame_length] ** 2)
        y[i:min(i+frame_length, siglen)] = E[j]
        j = j + 1
    E = smooth(E, window_len=5 )
    y = smooth(y, window_len = hop_size*2)
    y = y[:siglen]
    return y, E


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def nextpow2(m):
    """
    Calculating the next power of 2 for a given number m
    """
    y = int(2**(np.ceil(np.log2(m))))
    return(y)

def medclip(X, fctr, floordB = 0):
    """
    medclip carried out a median clipping for a given input matrix X.
    The function assigned a minimun predefined value for entries with values
    smaller or equal to a factor multiplied by the max of row and column medians
    of each entry.
    Input arguments:
    ----------------
    X - input matrix, fctr - multiplication factor, floordB - if a dB scale is
        required.    
    Returns:
    --------
    Y - the matrix after median clipping.
    """
    Xmedcols = np.median(X,0) 
    Xmedrows = np.median(X,1)
    Y = np.zeros(X.shape) + floordB
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > fctr*max(Xmedcols[j], Xmedrows[i]):
                Y[i,j] = X[i,j]
    return Y

def blobRemove(M,neigh_num = 2, floordB = -80, Thresh = -60 ):
    """
    Removing of isolated cells in a spectrogram (or mel-spectrogram) based on
    the number of neighbors of each cell. The spectrogram could be in a dB or a 
    linear scale. Please note that for the latter the default parameters should
    be modified.
    Input  arguments: 
    ----------------- 
        M - original spectrogram (image), neigh_num - number of neighbors, 
        floordB - value if number of neighbors<neigh_num. The default is -80.
        Thresh - threshold above which  1's are assigned in a binary matrix.
            The default is -60. 
    Returns
    -------
    M1 : the original matrix after removal of isolated cells (entries).
    """
    X = np.zeros(M.shape)
    X[M>Thresh] = 1
    M1 = np.copy(M)
    Mup = X[0:-2,1:-1]
    Mdown = X[2:,1:-1]
    Mleft = X[1:-1,0:-2]
    Mright = X[1:-1,2:]
    Mul = X[0:-2,0:-2]
    Mur = X[0:-2, 2:]
    Mdl = X[2:, 0:-2]
    Mdr = X[2:, 2:]
    Mneigh = Mup + Mdown + Mleft + Mright + Mul + Mur + Mdl + Mdr
    M2 = M1[1:-1,1:-1]
    M2[Mneigh<neigh_num] = floordB
    M1[1:-1,1:-1] = M2
    return M1
def small_objects_remove(M,floordB = -80, Thresh = -60, dsize = 10):   
    """
    small_objects_remove uses morphological image processing function
    (tophat) to remove small objects from a spectrogram
    (or any other gray scale image)
    Input arguments: M - input spectrogram or mel-scale spectrogram
    Returns: M1 - the spectrogram after cleaning.
    """
    from skimage import data
    from skimage import color, morphology
    
    X = np.zeros(M.shape)
    M1 = np.copy(M)
    X[M1>Thresh] = 1 # converting to a binary image
    selem =  morphology.disk(dsize)
    res = morphology.white_tophat(X, selem)
    M1[res == 0] = floordB
    # M = morphology.remove_small_objects(M, min_size=5, connectivity=1)
    
    return M1


def clean_small_objects(M,floordB = -80, Thresh = -60, min_size = 5, connectivity = 1):   
    """
    small_objects_remove uses morphological image processing function
    remove_small_objects (from scikit image) to remove small objects from a spectrogram
    (or any other gray scale image)
    Input arguments: M - input spectrogram or mel-scale spectrogram
    Returns: M1 - the spectrogram after cleaning.
    """
    from skimage import data
    from skimage import color, morphology
    
    X = np.zeros(M.shape)
    M1 = np.copy(M)
    X[M1>Thresh] = 1 # converting to a binary image
    X = X.astype(bool)
    res = morphology.remove_small_objects(X, min_size, connectivity)
    M1[res == 0] = floordB
    # M = morphology.remove_small_objects(M, min_size=5, connectivity=1)
    
    return M1

def pad_sig(x, Len, method = 1):   
    """
    ypad for padding an input signal or vector with zeros or random white
    Gaussian noise. The original signal is set in the middle of the padded sig
    Inputs: y - the input signal or vector.
            Len - required padded signal length
            method = 0 - zero padding, 1 - small white Gaussian noise
    Return - the padded signal (vector) 
    """
    eps = 1e-14
    
    if method == 0:
        xpad = np.zeros(Len)
        mid_start = Len // 2 - x.size // 2
        xpad[mid_start:mid_start + x.size] = x
    
    elif method == 1:
        xpad = eps * np.random.randn(Len)
        mid_start = Len // 2 - x.size // 2
        xpad[mid_start:mid_start + x.size] = x
    y = xpad
            
    return y

    
    

            
                


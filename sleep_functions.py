#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from scipy.fft import rfft, rfftfreq
import scipy.spatial.transform._rotation_groups
from scipy.integrate import simps
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def compute_power(data, window=10, samplerate = 1000, freq_limit=40):
    '''
    Computes a power spectrum from raw eeg/emg data.
    
    INPUTS:
    data = dataframe with 'eeg' and 'emg' as columns
    window = window of time (in seconds) that you want to compute power spectrum for. Default=10
    samplerate = samplerate at which data was collected, default = 1000
    freq_limit = maximum frequency (in Hz) that you want retained in the power spectrum. Default=40
    _______________________________________________________________________________________________

    OUTPUTS:
    eeg_power = dictionary wherein keys are epochs and values are power spectra for EEG
    emg_power = dictionary wherein keys are epochs and values are power spectra for EMG
    EMG_amp = dictionary wherein keys are epochs and values are relative EEG amplitude versus baseline
    EMG_amp = dictionary wherein keys are epoch and values are relative EMG amplitude versus baseline
    '''
    
    data_window = window * samplerate
    power_len = freq_limit * 10
    dt = 1/samplerate
    
    epoch=0
    eeg_power = {}
    emg_power = {}
    EEG_amp = {}
    EMG_amp = {}
    
    #Compute baseline EEG/EMG amplitudes for 30 sec (skipping first 10s)
    wake_EEG = np.mean(np.absolute(data['eeg'][10000:40000]))
    wake_EMG = np.mean(np.absolute(data['emg'][10000:40000]))
    
    for value in range(0,len(data),data_window):
        start = value
        end = start + data_window
        
        #Select EEG and EMG data based on start and stop points
        EEG = np.array(data['eeg'][start:end])
        EMG = np.array(data['emg'][start:end])
        
        #Calculate EMG_avg and EMG_max
        EEG_amp[epoch] = np.mean(np.absolute(EEG)) / wake_EEG
        EMG_amp[epoch] = np.max(np.absolute(EMG)) / wake_EMG
        
        #Compute Fourier Transform
        eeg_xf = rfft(EEG-EEG.mean())
        emg_xf = rfft(EMG-EMG.mean())
        
        #Compute power spectrum
        eeg_Sxx = (2 * dt**2 / 10 * (eeg_xf*eeg_xf.conj())).real
        eeg_Sxx = eeg_Sxx[0:power_len]
        
        emg_Sxx = (2 * dt**2 / 10 * (emg_xf*emg_xf.conj())).real
        emg_Sxx = emg_Sxx[0:power_len]

        #add eeg and emg power spectra to dictionary
        eeg_power[epoch] = eeg_Sxx
        emg_power[epoch] = emg_Sxx
        
        epoch += 1
    return(eeg_power, emg_power, EEG_amp, EMG_amp)


# In[1]:


#Smooth Sxx function
def smooth_signal(eeg_power, emg_power, window_len=4, window_type='flat'):
    '''
    Smooths the power spectrum outputted by the compute_power function using a moving window.
    
    INPUTS:
    eeg_power = dictionary wherein keys are epochs and values are power spectra for EEG
    emg_power = dictionary wherein keys are epochs and values are power spectra for EMG
    window_len = size of smoothing window. Bigger numbers mean more smoothing. Default = 4
    window_type = kind of smoothing function to be utilized. Default = 'flat'
    _______________________________________________________________________________________
    
    OUTPUTS:
    smoothed_eeg_dict = dictionary wherein keys are epochs and values are smoothed power spectra for EEG
    smoothed_emg_dict = dictionary wherein keys are epochs and values are smoothed power spectra for EMG
    '''
    
    smoothed_eeg = {}
    smoothed_emg = {}
    
    if window_len % 2 != 0:
        raise ValueError('window length must be an even integer')
    
    for epoch in eeg_power:
        x_eeg = eeg_power[epoch]
        x_emg = emg_power[epoch]
        if x_eeg.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")

        if x_eeg.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")

        if window_len<3:
            return x_eeg, x_emg

        if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise(ValueError, "Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s_eeg=np.r_[x_eeg[window_len-1:0:-1],x_eeg,x_eeg[-2:-window_len-1:-1]]
        s_emg=np.r_[x_emg[window_len-1:0:-1],x_emg,x_emg[-2:-window_len-1:-1]]

        if window_type == 'flat': # Moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window_type+'(window_len)')

        y_eeg=np.convolve(w/w.sum(),s_eeg,mode='valid')
        y_emg=np.convolve(w/w.sum(),s_emg,mode='valid')

        smoothed_eeg[epoch] = y_eeg[(int(window_len/2)-1):-int(window_len/2)]
        smoothed_emg[epoch] = y_emg[(int(window_len/2)-1):-int(window_len/2)]
    return(smoothed_eeg, smoothed_emg)


# In[1]:


def compute_metrics(smoothed_eeg, freq_res = 0.1):
    '''
    Computes a set of metrics from smoothed_power_dict.
    
    INPUTS:
    smoothed_eeg_dict = dictionary wherein keys are epochs and value are smoothed power spectra for EEG
    smoothed_emg_dict = dictionary wherein keys are epochs and value are smoothed power spectra for EMG
    freq_res = frequency resolution of power spectrum. Default = 0.1
    _____________________________________________________________________________________________
    
    OUTPUTS:
    metrics = Dictionary where keys are epochs and lists contain the following computed metrics:
        [0] = delta_power = proportion of power falling in delta band (defined in idx_delta)
        [1] = theta_power = proportion of power falling in theta band (defined in idx_theta)
    '''
    metrics = {}
    
    for epoch in smoothed_eeg:
        x = smoothed_eeg[epoch]
        metric_list = []
        
        power_axis = np.arange(len(x))*freq_res
        idx_delta = np.logical_and(power_axis >= 0.5, power_axis <= 4)
        idx_theta = np.logical_and(power_axis >= 5.5, power_axis <= 8.5)
        
        #compute total power
        total_power = simps(x, dx=freq_res)
        
        #compute delta power
        delta_power = (simps(x[idx_delta], dx=freq_res)) / total_power
        metric_list.append(delta_power)
        
        #compute theta power
        theta_power = (simps(x[idx_theta], dx=freq_res)) / total_power
        metric_list.append(theta_power)
        
        metrics[epoch] = metric_list
    return(metrics)
            


# In[1]:


def plot_confusion_matrix(y,y_predict,label_list):
    "this function plots the confusion matrix and outputs the values of the matrix as an array, cm"

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(label_list); ax.yaxis.set_ticklabels(label_list)
    return(cm)


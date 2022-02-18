#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from scipy.fft import rfft, rfftfreq
import scipy.spatial.transform._rotation_groups  # keep this for compiling
from scipy.integrate import simps
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def compute_power(data, window: int = 10, samplerate: int = 1000, freq_limit: int = 50) -> tuple[dict[int, float],
                                                                                                 dict[int, float]]:
    """
    Computes a power spectrum from raw eeg/emg data.

    INPUTS:
    data = dataframe with 'eeg' and 'emg' as columns
    window = window of time (in seconds) that you want to compute power spectrum for. Default=10
    freq_limit = maximum frequency (in Hz) that you want retained in the power spectrum. Default=50
    _______________________________________________________________________________________________

    OUTPUTS:
    eeg_power = dictionary wherein keys are epochs and values are power spectra for EEG
    emg_power = dictionary wherein keys are epochs and values are power spectra for EMG
    """

    # array length = window length * samplerate
    array_len = window * samplerate
    freq_limit = freq_limit * 10
    dt = 1 / samplerate

    eeg_power = {}
    emg_power = {}

    epoch = 0
    for value in range(0, len(data), array_len):
        start = value
        end = start + array_len

        # Select EEG and EMG data based on start and stop points
        EEG = data['eeg'][start:end]
        EMG = data['emg'][start:end]

        # Compute Fourier Transform
        eeg_xf = rfft(np.array(EEG) - np.array(EEG).mean())
        emg_xf = rfft(np.array(EMG) - np.array(EMG).mean())

        # Compute power spectrum
        eeg_Sxx = (2 * dt ** 2 / window * (eeg_xf * eeg_xf.conj())).real
        eeg_Sxx = eeg_Sxx[0:freq_limit]

        emg_Sxx = (2 * dt ** 2 / window * (emg_xf * emg_xf.conj())).real
        emg_Sxx = emg_Sxx[0:freq_limit]

        eeg_power[epoch] = eeg_Sxx
        emg_power[epoch] = emg_Sxx

        epoch += 1

    return eeg_power, emg_power


# Smooth Sxx function
def smooth_signal(eeg_power: dict[int, float], emg_power: dict[int, float], window_len: int = 4,
                  smooth_type: str = 'flat') -> tuple[dict[int, float], dict[int, float]]:
    """
    Smooths the power spectrum outputted by the compute_power function using a moving window.

    INPUTS:
    eeg_power = dictionary wherein keys are epochs and values are power spectra for EEG
    emg_power = dictionary wherein keys are epochs and values are power spectra for EMG
    window_len = size of smoothing window. Bigger numbers mean more smoothing. Default = 4
    smooth_type = kind of smoothing function to be utilized. Default = 'flat'
    _______________________________________________________________________________________

    OUTPUTS:
    smoothed_eeg = dictionary wherein keys are epochs and value are smoothed power spectra for EEG
    smoothed_emg = dictionary wherein keys are epochs and value are smoothed power spectra for EMG
    """

    smoothed_eeg = {}
    smoothed_emg = {}

    if window_len % 2 != 0:
        raise ValueError('window length must be an even integer')

    for epoch in eeg_power:
        x_eeg = eeg_power[epoch]
        x_emg = emg_power[epoch]
        if x_eeg.ndim != 1:
            raise (ValueError, "smooth only accepts 1 dimension arrays.")

        if x_eeg.size < window_len:
            raise (ValueError, "Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x_eeg, x_emg

        if smooth_type not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise (ValueError, "Smooth_type must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s_eeg = np.r_[x_eeg[window_len - 1:0:-1], x_eeg, x_eeg[-2:-window_len - 1:-1]]
        s_emg = np.r_[x_emg[window_len - 1:0:-1], x_emg, x_emg[-2:-window_len - 1:-1]]

        if smooth_type == 'flat':  # Moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + smooth_type + '(window_len)')

        y_eeg = np.convolve(w / w.sum(), s_eeg, mode='valid')
        y_emg = np.convolve(w / w.sum(), s_emg, mode='valid')

        smoothed_eeg[epoch] = y_eeg[(int(window_len / 2) - 1):-int(window_len / 2)]
        smoothed_emg[epoch] = y_emg[(int(window_len / 2) - 1):-int(window_len / 2)]
    return smoothed_eeg, smoothed_emg


def compute_relative_power(smoothed_eeg: dict[int, float], freq_res: float = 0.1, delta_lower: float = 0.5,
                           delta_upper: float = 4, theta_lower: float = 5.5,
                           theta_upper: float = 8.5) -> pd.DataFrame:
    """
    Computes relative power for delta and theta bands.

    INPUTS:
    smoothed_eeg_dict = dictionary wherein keys are epochs and value are smoothed power spectra for EEG
    freq_res = frequency resolution of power spectrum. Default = 0.1
    delta_lower = lower limit of delta band
    delta_upper = upper limit of delta band
    theta_lower = lower limit of theta band
    theta_upper = upper limit of theta band
    _____________________________________________________________________________________________

    OUTPUTS:
    rel_power = a pd.dataframe where rows are epochs and columns are as follows:
        delta_power = proportion of power falling in delta band for epoch x
        theta_power = proportion of power falling in theta band for epoch x
        theta_over_delta = ratio of theta power to delta power for epoch x
    """
    rel_power = {}

    for epoch in smoothed_eeg:
        x = smoothed_eeg[epoch]
        rel_power_list = []

        power_axis = np.arange(len(x)) * freq_res
        idx_delta = np.logical_and(power_axis >= delta_lower, power_axis <= delta_upper)
        idx_theta = np.logical_and(power_axis >= theta_lower, power_axis <= theta_upper)

        # compute total power
        total_power = simps(x, dx=freq_res)

        # compute delta power
        delta_power = (simps(x[idx_delta], dx=freq_res)) / total_power
        rel_power_list.append(delta_power)

        # compute theta power
        theta_power = (simps(x[idx_theta], dx=freq_res)) / total_power
        rel_power_list.append(theta_power)

        # compute proportion of theta to delta
        theta_over_delta = theta_power / delta_power
        rel_power_list.append(theta_over_delta)

        rel_power[epoch] = rel_power_list

    rel_power = pd.DataFrame.from_dict(rel_power).T
    rel_power.columns = ['delta_rel', 'theta_rel', 'theta_over_delta']

    return rel_power


def compute_signal_features(data: pd.DataFrame, window: int = 10, samplerate: int = 1000, start_epoch: int = 9) \
        -> pd.DataFrame:
    """
    Computes features from EEG and EMG data for ML model

    INPUTS:
    data = dataframe with 'eeg' and 'emg' as columns
    start_epoch = epoch for which baseline wake EEG/EMG measures will be calculated
    _______________________________________________________________________________________________

    OUTPUTS:
    output = a pd.DataFrame where rows are epochs and columns are as follows:
        EEG_std = dictionary,keys are epochs and values are relative SD of this epoch versus base_EEG
        EEG_ss = dictionary, keys are epochs and values are sum of squared for this epoch compared to base_EEG
        EEG_ amp = dictionary, keys are epochs and values are relative amplitidue of this epoch compare to base_EEG
        EMG_std = dictionary, keys are epochs and values are relative SD of this epoch versus base_EMG
        EEG_ss = dictionary, keys are epochs and values are sum of squared for this epoch compared to base_EMG
        EMG_events = dictionary, keys are epochs and values are number of EMG events within that epoch
    """

    array_len = window * samplerate

    EEG_std = {}
    EEG_ss = {}
    EEG_amp = {}

    EMG_std = {}
    EMG_ss = {}
    EMG_events = {}

    # Compute baseline EEG/EMG amplitudes and EMG event threshold
    base_std_EEG = np.std(data['eeg'][(start_epoch * array_len):((start_epoch * array_len) + array_len)])
    base_amp_EEG = np.mean(np.absolute(data['eeg'][(start_epoch * array_len):((start_epoch * array_len) + array_len)]))
    base_ss_EEG = np.sum(np.square(data['eeg'][(start_epoch * array_len):((start_epoch * array_len) + array_len)]))

    base_std_EMG = np.std(data['emg'][(start_epoch * array_len):((start_epoch * array_len) + array_len)])
    base_ss_EMG = np.sum(np.square(data['emg'][(start_epoch * array_len):((start_epoch * array_len) + array_len)]))

    event_threshold = 2 * base_std_EMG

    epoch = 0
    for value in range(0, len(data), array_len):
        start = value
        end = start + array_len

        # Select EEG and EMG data based on start and stop points
        EEG = data['eeg'][start:end]
        EMG = data['emg'][start:end]

        # Calculate EEG_std, EMG_std, and EEG_amp, and add to dictionaries
        EEG_std[epoch] = np.std(EEG) / base_std_EEG
        EEG_amp[epoch] = np.mean(np.absolute(EEG)) / base_amp_EEG
        EEG_ss[epoch] = np.sum(np.square(EEG)) / base_ss_EEG

        EMG_ss[epoch] = np.sum(np.square(EMG)) / base_ss_EMG
        EMG_std[epoch] = np.std(EMG) / base_std_EMG

        # Calculate EMG events above event_treshold
        event_array = EMG.loc[EMG > event_threshold]

        if not len(event_array) == 0:
            event_dict = {}
            count = 0
            first = event_array.index[0]

            for idx in event_array.index:
                if idx + 1 not in event_array.index:
                    last = idx
                    event_dict[count] = event_array.loc[first:last]
                    count += 1
                    first = last
            EMG_events[epoch] = len(event_dict)
        else:
            EMG_events[epoch] = 0

        epoch += 1

    output = pd.DataFrame([EEG_std, EEG_ss, EEG_amp, EMG_std, EMG_ss, EMG_events]).T
    output.columns = ['EEG_std', 'EEG_ss', 'EEG_amp', 'EMG_std', 'EMG_ss', 'EMG_events']

    return output


def generate_features(data: pd.DataFrame, scores: pd.DataFrame, start_epoch: int = 9) -> pd.DataFrame:
    """
    Run all functions to compute power spectrum and feature for ML in one dataframe

    INPUTS:
    data = dataframe with 'eeg' and 'emg' as columns
    scores = dataframe containing scores ['Wake', 'Non REM', 'REM', 'Unscored']
    start_epoch = epoch in data to use for normalization of other epochs
    __________________________________________________________________________________________________

    OUTPUTS:
    metrics = dataframe where columns are features and rows are metrics for those features calculated for each epoch

    """

    eeg_power, emg_power = compute_power(data)
    smoothed_eeg, smoothed_emg = smooth_signal(eeg_power, emg_power)

    signal_features = compute_signal_features(data, start_epoch=start_epoch)
    rel_power = compute_relative_power(smoothed_eeg)

    metrics = signal_features.merge(rel_power, left_on=signal_features.index, right_on=rel_power.index, how='outer')

    # get rid of Key_0 column from merge
    metrics = metrics.iloc[:, 1:]

    try:
        scores.rename(columns={' Score': 'Score'}, inplace=True)
        metrics['score'] = scores['Score']
    except KeyError:
        metrics['score'] = scores['score']

    return metrics


def plot_confusion_matrix(y: np.ndarray, y_predict: np.ndarray, label_list: list) -> np.ndarray:
    "this function plots the confusion matrix"

    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='.0f')  # annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(label_list);
    ax.yaxis.set_ticklabels(label_list)
    return cm


def modify_scores(pred: list) -> list:
    ls = []
    for idx, val in enumerate(pred):
        try:
            if pred[idx - 1] == 'Wake' and val == 'REM':
                ls.append('Wake')
            elif (pred[idx - 2] == pred[idx - 1] == pred[idx + 1] == pred[idx + 2] != val) & (val != 'REM'):
                ls.append(pred[idx - 1])
            else:
                ls.append(val)
        except IndexError:
            ls.append(val)
            next
    return ls

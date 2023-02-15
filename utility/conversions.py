import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import scipy.signal as signal
from statistics import median
from scipy.interpolate import interp1d
import torch
from torchaudio.functional import bandpass_biquad, lowpass_biquad
import math
import pandas as pd
from sklearn.model_selection import train_test_split


def minmax_scale(time_series, axis=0, max_only=False):
    """
    Scale ND-arrays along axis so the minimum is at 0, maximum is at 1. Works for multichannel emg or mocap, or single
    channel labels too.
    :return: Array normalized along given axis
    """
    if max_only:
        mins = 0
    else:
        mins = np.min(time_series, axis=axis)
    maxs = np.max(time_series, axis=axis)
    return (time_series - np.expand_dims(mins, axis=axis)) / (np.expand_dims(maxs, axis=axis)-np.expand_dims(mins, axis=axis))


def envelope(bipolar_data, fs=2000, hib=20, lob=400, lop=10, axis=0):
    hi_band = hib / (fs / 2)
    lo_band = lob / (fs / 2)
    b, a = signal.butter(4, Wn=[hi_band, lo_band], btype='bandpass', output='ba')
    filtered_data = signal.lfilter(b, a, bipolar_data, axis=axis)  # 2nd order bandpass butterworth filter
    filtered_data = abs(filtered_data)  # Rectify the signal
    lo_pass = lop / (fs / 2)
    b, a = signal.butter(4, lo_pass, output='ba')  # create low-pass filter to get EMG envelope
    filtered_data = signal.lfilter(b, a, filtered_data, axis=axis)
    return filtered_data


def linear_temporal_normalisation(start_indices, time_series, axis=0, minmax=False):
    repetitions = np.split(time_series, start_indices, axis=axis)
    max_length = max([rep.shape[axis] for rep in repetitions])
    rep_len = [len(rep) for rep in repetitions]
    repetitions = [rep for rep in repetitions if len(rep) < median(rep_len)*1.5]
    new_idx = np.arange(0, max_length)
    warped_repetitions = np.zeros((max_length, time_series.shape[1], len(repetitions)))
    for i in range(len(repetitions)):
        warped = []
        rep_length, channels = repetitions[i].shape
        old_idx = np.linspace(0, max_length, rep_length)
        for j in range(channels):
            warped_sig = np.interp(new_idx, old_idx, repetitions[i][:, j])
            warped.append(warped_sig)
        warped = np.array(warped).T
        warped_repetitions[:, :, i] = warped
    if minmax is True:
        norm_warped = minmax_scale(warped_repetitions, axis=0)
        warped_repetitions = norm_warped
    return warped_repetitions


def upsample_based_on_axis(time_series, n_samples, axis=0):
    if time_series.shape[axis] != n_samples:
        xnew = np.linspace(0, time_series.shape[axis]-1, num=n_samples)
        xold = np.arange(0, time_series.shape[0])
        upsampled_series = interp1d(xold, time_series, axis=axis)(xnew)
        return upsampled_series


def flatten_channels(bipolar_signals, labels):
    # recall that the time axis is 0
    n_channels = bipolar_signals.shape[-1]
    labels = np.tile(labels.T, n_channels).T
    bipolar_signals = np.expand_dims(bipolar_signals.flatten('F'), axis=1)
    return bipolar_signals, labels


def split_signals_into_TCN_windows(signals, labels, window_length, window_step, label_delay=0, reps_first=True):
    signal_snippets = []
    label_snippets = []
    for i in range(0, int(signals.shape[0] - window_length - 1- label_delay), window_step):
        signal_snippets.append(np.expand_dims(signals[i: i + window_length, :], axis=2))
        label_snippets.append(labels[i + window_length + 1 + label_delay, :])
    windowed_signals = np.concatenate(signal_snippets, axis=2)#.transpose((2, 1, 0))
    windowed_labels = np.array(label_snippets).transpose((1, 0))
    if reps_first:
        windowed_signals = windowed_signals.transpose((2, 0, 1))
        windowed_labels = windowed_labels.transpose((1, 0))
    return windowed_signals, windowed_labels


def group_windows_into_sequences(signals, labels, n_windows_per_sequence, window_axis=-1):
    cropped_n_windows = math.floor(signals.shape[window_axis] / n_windows_per_sequence)
    signals = signals[:, :, 0: cropped_n_windows*n_windows_per_sequence]
    signals = signals.reshape((signals.shape[0], signals.shape[1], n_windows_per_sequence, -1), order='F').transpose((2, 0, 1, 3))
    labels = labels[:, 0: cropped_n_windows*n_windows_per_sequence].reshape((n_windows_per_sequence, labels.shape[0], -1), order='F')
    return signals, labels


def split_into_batches(x_train, y_train, x_test, y_test, batch_size, batch_axis=1):
    if x_train.ndim == 4:
        cropped_length = int(x_train.shape[-1] - (x_train.shape[-1] % batch_size))
        x_train = x_train[:, :, :, :cropped_length]
        y_train = y_train[:, :, :cropped_length]
        x_batches = []
        for i in range(0, int(x_train.shape[-1]), int(batch_size)):
            x_batches.append(np.expand_dims(x_train[:, :, :, i:i+batch_size], axis=3))
        x_train = np.concatenate(x_batches, axis=3)
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], -1, batch_size))
        x_transposition = [0, 1, 2, 3]
        x_transposition.insert(int(batch_axis), 4)
        y_transposition = [0, 1, 2]
        y_transposition.insert(int(batch_axis), 3)
        x_train = x_train.transpose(x_transposition)
        y_train = y_train.transpose(y_transposition)
        x_test = np.expand_dims(x_test, axis=batch_axis)
        y_test = np.expand_dims(y_test, axis=batch_axis)
    elif x_train.ndim == 3:
        cropped_length = int(x_train.shape[-1] - (x_train.shape[-1] % batch_size))
        x_train = x_train[:, :, :cropped_length]
        y_train = y_train[:, :cropped_length]
        x_batches = []
        for i in range(0, int(x_train.shape[-1]), int(batch_size)):
            x_batches.append(np.expand_dims(x_train[:, :, i:i + batch_size], axis=2))
        x_train = np.concatenate(x_batches, axis=2)
        y_train = y_train.reshape((y_train.shape[0], -1, batch_size))
        x_transposition = [0, 1, 2]
        x_transposition.insert(int(batch_axis), 3)
        y_transposition = [0, 1]
        y_transposition.insert(int(batch_axis), 2)
        x_train = x_train.transpose(x_transposition)
        y_train = y_train.transpose(y_transposition)
        x_test = np.expand_dims(x_test, axis=batch_axis)
        y_test = np.expand_dims(y_test, axis=batch_axis)
    else:
        raise ValueError('The input dimensions of the array are wrong')
    return x_train, y_train, x_test, y_test


def shuffle(signals, labels):
    shuffler = np.random.permutation(signals.shape[-1])
    if signals.ndim == 4:
        signals = signals[:, :, :, shuffler]
        labels = labels[:, :, shuffler]
    elif signals.ndim == 3:
        signals = signals[:, :, shuffler]
        labels = labels[:, shuffler]
    elif signals.ndim == 5:
        signals = signals[:, :, :, :, shuffler]
        labels = labels[:, :, :, shuffler]
    return signals, labels


def split_into_train_test(signals, labels, train_size, split_axis):
    if signals.ndim == 4:
        if split_axis == -1:
            signals = signals.transpose((3, 0, 1, 2))
            labels = labels.transpose((2, 0, 1))
        elif split_axis != -1 and split_axis!= 0:
            raise ValueError('You need to have the split axis be either in first or last place.')
        x_train, x_test, y_train, y_test = train_test_split(signals, labels, train_size=train_size, shuffle=False)
        x_train = x_train.transpose((1, 2, 3, 0))
        x_test = x_test.transpose((1, 2, 3, 0))
        y_train = y_train.transpose((1, 2, 0))
        y_test = y_test.transpose((1, 2, 0))
    elif signals.ndim == 3:
        if split_axis == -1:
            signals = signals.transpose((2, 0, 1))
            labels = labels.transpose((1, 0))
        elif split_axis != -1 and split_axis!= 0:
            raise ValueError('You need to have the split axis be either in first or last place.')
        x_train, x_test, y_train, y_test = train_test_split(signals, labels, train_size=train_size, shuffle=False)
        x_train = x_train.transpose((2, 1, 0))
        x_test = x_test.transpose((2, 1, 0))
        y_train = y_train.transpose((1, 0))
        y_test = y_test.transpose((1, 0))
    elif signals.ndim == 2:
        if split_axis == -1:
            signals = signals.transpose((2, 0, 1))
            labels = labels.transpose((1, 0))
        x_train, x_test, y_train, y_test = train_test_split(signals, labels, train_size=train_size, shuffle=False)
    else:
        raise ValueError('Your input arrays have the wrong shape.')
    return x_train, x_test, y_train, y_test
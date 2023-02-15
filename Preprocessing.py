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
import statistics
import math
from scipy import stats
from math import prod
from utility.conversions import upsample_based_on_axis, flatten_channels, envelope, linear_temporal_normalisation, \
    shuffle, split_signals_into_TCN_windows, split_into_batches, split_into_train_test, group_windows_into_sequences


class Preprocessing:
    def __init__(self, hdemg_data, data_labels, patch_shape=(13, 5), facing_down=True, z_threshold=2, outlier_process='filter',
                 central_roi_shape=None):
        self.data = hdemg_data
        self.facing_down = facing_down
        self.patch_shape = patch_shape
        self.z_threshold = z_threshold
        self.central_roi_shape = central_roi_shape
        self.outlier_process = outlier_process
        self.labels = data_labels.T
        self.rms = None
        self.patch_idx = None
        self.outlier_channels = None
        self.channels = None # This gives us a list of the channels that should be included
        self.bipolar_pairs = None
        self.bipolar_signals = None
        self.complete_data = None

        self.snake_grid()
        self.find_outliers()
        self.filter_outliers()

        if central_roi_shape:
            self.central_roi_channels(roi_shape=self.central_roi_shape)
            # if outlier_process == 'filter':
            #     self.filter_outliers()
            # elif outlier_process == 'remove':
            #     print('Before:', self.channels)
            #     self.channels = [channel for channel in self.channels if channel not in self.outlier_channels]
            #     print('After:', self.channels)
            self.get_bipolar_combinations_from_channels()

        self.concatenate_data()

    def snake_grid(self):
        n_electrodes = prod(self.patch_shape)
        idx = np.arange(n_electrodes).reshape((self.patch_shape[1], self.patch_shape[0]))
        idx[1::2] = idx[1::2, ::-1]
        idx = idx.T
        if self.facing_down:
            self.patch_idx = idx
        else:
            self.patch_idx = np.fliplr(np.flipud(idx))

    def find_outliers(self):
        rms = []
        data = self.data
        for i in range(data.shape[1]):
            rms.append(np.sqrt(np.mean(data[:, i] ** 2)))

        self.rms = rms
        z = np.abs(stats.zscore(rms))
        threshold = self.z_threshold

        outlier_channels = []
        for i in range(len(z)):
            if z[i] > threshold:
                outlier_channels.append(i)

        self.outlier_channels = outlier_channels

    def filter_outliers(self):
        data = self.data
        for i in range(len(self.outlier_channels)):
            channel_idx_i, channel_idx_j = np.where(self.patch_idx == self.outlier_channels[i])
            channel_idx_i = int(channel_idx_i)
            channel_idx_j = int(channel_idx_j)
            neighbour_idx = [self.patch_idx[channel_i_range, channel_j_range]
                             for channel_i_range in range(channel_idx_i-1, channel_idx_i+2)
                             for channel_j_range in range(channel_idx_j-1, channel_idx_j+2)
                             if (0 <= channel_i_range < self.patch_idx.shape[0]
                                 and 0 <= channel_j_range < self.patch_idx.shape[1]
                                 and (channel_i_range != channel_idx_i
                                      or channel_j_range != channel_idx_j))]
            for sample in range(data.shape[0]):
                new_value = statistics.median([data[sample, val-1] for val in neighbour_idx if val != 0])
                data[sample, self.outlier_channels[i]] = new_value
        self.data = data

    def central_roi_channels(self, roi_shape=(5, 5)):
        central_i = math.floor(self.patch_shape[0]/2)
        central_j = math.floor(self.patch_shape[1]/2)
        central_roi = [self.patch_idx[i_range, j_range]
                       for i_range in range(central_i-math.floor(roi_shape[0]/2), central_i+math.ceil(roi_shape[0]/2))
                       for j_range in range(central_j-math.floor(roi_shape[1]/2), central_j+math.ceil(roi_shape[1]/2))]
        self.channels = central_roi

    def get_bipolar_combinations_from_channels(self, max_step_over=2, max_step_across=1):
        all_combinations = []
        for i in range(len(self.channels)):
            for j in range(len(self.channels)):
                # ADD A LIMIT FOR DISTANCE!
                if i != j:
                    xi, yi = np.where(self.patch_idx == self.channels[i])
                    xj, yj = np.where(self.patch_idx == self.channels[j])
                    if abs(xi[0] - xj[0]) < (max_step_over + 1) and abs(yi[0] - yj[0]) < (max_step_across + 1) \
                            and xi[0] != xj[0]:
                        if i < j:
                            all_combinations.append((self.channels[i], self.channels[j]))
                        elif j < i:
                            all_combinations.append((self.channels[j], self.channels[i]))
        all_combinations = list(set(all_combinations))
        self.bipolar_pairs = all_combinations
        bipolar_signals = []
        for channel_pair in self.bipolar_pairs:
            ch1, ch2 = channel_pair
            bipolar_signals.append(self.data[:, ch1-1] - self.data[:, ch2-1])
        self.bipolar_signals = np.array(bipolar_signals).T

    def concatenate_data(self):
        assert self.bipolar_signals.shape[0] == self.labels.shape[0]
        # labels = np.tile(self.labels, (self.bipolar_signals.shape[0], 0))
        # data = self.bipolar_signals.reshape((-1, 1))
        # labels = labels.reshape((-1, 1))
        self.complete_data = np.concatenate((self.bipolar_signals, self.labels), axis=1)


def get_hdemg_data(subject, muscle, state, hdemg_roi=(3, 1), flatten=True, normalise=True, interpolation_factor=None):
    data_path = str('/media/ag6016/Storage/AIM2022/AllSubjects/' + subject + '_csv/' + subject + '_' + muscle + '_' +
                    state + '.csv')
    print(data_path)
    data = pd.read_csv(data_path, delimiter=';').values
    hdemg_data = data[:, :-1]
    labels = np.expand_dims(data[:, -1], axis=0)
    processed_data = Preprocessing(hdemg_data, labels, central_roi_shape=hdemg_roi, outlier_process='remove')
    EMG_signals = processed_data.bipolar_signals  # (229073, 21)
    knee_angles = processed_data.labels  # (229073, 1)
    if flatten:
        EMG_signals, knee_angles = flatten_channels(EMG_signals, knee_angles)
    if normalise:
        EMG_signals = (EMG_signals - EMG_signals.mean(axis=0))
        for channel in range(EMG_signals.shape[-1]):
            EMG_signals[:, channel] = EMG_signals[:, channel] / (0.95 * np.max(np.abs(EMG_signals[:, channel])))
    if interpolation_factor is not None:
        EMG_signals = upsample_based_on_axis(EMG_signals, int(EMG_signals.shape[0]*interpolation_factor), axis=0)
        knee_angles = upsample_based_on_axis(knee_angles, int(knee_angles.shape[0] * interpolation_factor), axis=0)
    assert len(EMG_signals) == len(knee_angles)
    return EMG_signals, knee_angles


def get_delsys_data(muscle, normalise=True, interpolation_factor=None):
    data = np.load('/home/ag6016/Desktop/AIM2022/Bipolar_full_data.npy')
    muscle_list = np.array(['Quad', 'Ham', 'Tibialis', 'Soleus'])
    EMG_signals = np.expand_dims(data[:, np.where(muscle_list == muscle)[0][0]], axis=1)
    knee_angles = np.expand_dims(data[:, -1], axis=1)
    if normalise:
        EMG_signals = (EMG_signals - EMG_signals.mean(axis=0))
        for channel in range(EMG_signals.shape[-1]):
            EMG_signals[:, channel] = EMG_signals[:, channel] / (0.95 * np.max(np.abs(EMG_signals[:, channel])))
    if interpolation_factor is not None:
        EMG_signals = upsample_based_on_axis(EMG_signals, int(EMG_signals.shape[0]*interpolation_factor), axis=0)
        knee_angles = upsample_based_on_axis(knee_angles, int(knee_angles.shape[0] * interpolation_factor), axis=0)
    assert len(EMG_signals) == len(knee_angles)
    return EMG_signals, knee_angles


def normalise_based_on_labels(signal, label, cycle_length=4000):
    label_peaks, _ = find_peaks(np.squeeze(label), distance=1000, height=0.6)
    warped_signal = linear_temporal_normalisation(label_peaks, signal)
    warped_label = linear_temporal_normalisation(label_peaks, label)
    warped_signal = upsample_based_on_axis(warped_signal, cycle_length, axis=0)
    warped_label = upsample_based_on_axis(warped_label, cycle_length, axis=0)
    signal = np.expand_dims(warped_signal.flatten('F'), axis=1)
    label = np.expand_dims(warped_label.flatten('F'), axis=1)
    return signal, label


def align_signals_together(hdemg_signal, hdemg_label, delsys_signal, delsys_label):
    hdemg_signal, hdemg_label = normalise_based_on_labels(hdemg_signal, hdemg_label, cycle_length=4000)
    delsys_signal, delsys_label = normalise_based_on_labels(delsys_signal, delsys_label, cycle_length=4000)
    hdemg_label_peaks, _ = find_peaks(np.squeeze(hdemg_label), distance=1000, height=0.6)
    delsys_label_peaks, _ = find_peaks(np.squeeze(delsys_label), distance=1000, height=0.6)
    hdemg_activation_time = []
    delsys_activation_time = []
    filtered_hdemg = envelope(hdemg_signal, lop=5)
    filtered_delsys = envelope(delsys_signal, lop=5)
    for i in range(len(hdemg_label_peaks)-1):
        max_hdemg = np.argmax(filtered_hdemg[hdemg_label_peaks[i]:hdemg_label_peaks[i+1], 0])
        hdemg_activation_time.append(max_hdemg)
    for i in range(len(delsys_label_peaks)-1):
        max_delsys = np.argmax(filtered_delsys[delsys_label_peaks[i]:delsys_label_peaks[i+1], 0])
        delsys_activation_time.append(max_delsys)
    hdemg_activation_time = statistics.median(hdemg_activation_time)
    delsys_activation_time = statistics.median(delsys_activation_time)
    # shift the signal
    difference = int(abs(hdemg_activation_time - delsys_activation_time))
    if hdemg_activation_time > delsys_activation_time:
        hdemg_signal = hdemg_signal[difference::, :]
        hdemg_label = hdemg_label[0:len(hdemg_signal), :]
        crop_to_peak = max([peak for peak in hdemg_label_peaks if peak < len(hdemg_signal)])
        hdemg_signal = hdemg_signal[0:crop_to_peak, :]
        hdemg_label = hdemg_label[0:crop_to_peak, :]
    else:
        delsys_signal = delsys_signal[difference::, :]
        delsys_label = delsys_label[0:len(delsys_signal), :]
        crop_to_peak = max([peak for peak in delsys_label_peaks if peak < len(delsys_signal)])
        delsys_signal = delsys_signal[0:crop_to_peak, :]
        delsys_label = delsys_label[0:crop_to_peak, :]

    # lengthen the target signal to match the length of the source signal
    if len(delsys_signal) < len(hdemg_signal):
        while len(delsys_signal) < len(hdemg_signal):
            delsys_signal = np.concatenate([delsys_signal, delsys_signal], axis=0)
            delsys_label = np.concatenate([delsys_label, delsys_label], axis=0)
    delsys_signal = delsys_signal[0:len(hdemg_signal), :]
    delsys_label = delsys_label[0:len(hdemg_signal), :]

    return hdemg_signal, hdemg_label, delsys_signal, delsys_label


class TCNDataPrep:
    def __init__(self, EMG_signals, labels, window_length, window_step, batch_size, sequence_length=10, label_delay=0,
                 training_size=0.9, lstm_sequences=True, split_data=True, shuffle_full_dataset=True):
        # self.EMG_signals = EMG_signals
        # self.labels = labels
        # self.n_channels = self.EMG_signals.shape[-1]
        self.window_length = window_length
        self.window_step = window_step
        self.prediction_delay = label_delay
        self.batch_size = batch_size
        self.sequence_length = sequence_length  # corresponds to the number of windows per sequence
        self.training_size = training_size

        average_values = []
        std_values = []
        for channel in range(EMG_signals.shape[-1]):
            average_values.append(np.mean(EMG_signals[:, channel]))
            std_values.append(np.std(EMG_signals[:, channel], dtype=np.float64))
        self.norm_values = average_values, std_values

        # WINDOW THE SIGNAL ----------------------------------------------------------------------------------------
        self.EMG_signals, self.labels = split_signals_into_TCN_windows(EMG_signals, labels, self.window_length,
                                                                       self.window_step, self.prediction_delay, False)

        if shuffle_full_dataset:
            # SHUFFLE ALONG THE N_SEQUENCE AXIS ------------------------------------------------------------------------
            self.EMG_signals, self.labels = shuffle(self.EMG_signals, self.labels)

        if split_data:  # ==============================================================================================
            # SPLIT INTO TRAIN-TEST ------------------------------------------------------------------------------------
            self.x_train, self.x_test, self.y_train, self.y_test = split_into_train_test\
                (self.EMG_signals, self.labels, train_size=self.training_size, split_axis=-1)
            # x train of shape (sequence_length, n_channels, window_length, n_training_sequences)
            # y train of shape (sequence_length, n_channels, n_training_sequences)
            # x test of shape (sequence_length, n_channels, window_length, n_testing_sequences)
            # y test of shape (sequence_length, n_channels, n_testing_sequences)

            # NORMALISE THE SIGNAL -------------------------------------------------------------------------------------
            # THIS NEEDS TO BE ONCE THE DATA HAS ALREADY BEEN SPLIT SO THAT THERE IS NO INFORMATION LEAKAGE BETWEEN THE
            # TRAINING AND TESTING SETS
            for channel in range(self.x_train.shape[0]):
                if np.all(self.x_train[channel, :, :]) != 0:
                    average = np.mean(self.x_train[channel, :, :])
                    std = np.std(self.x_train[channel, :, :], dtype=np.float64)
                    self.x_train[channel, :, :] = (self.x_train[channel, :, :] - average) / std
                    self.x_test[channel, :, :] = (self.x_test[channel, :, :] - average) / std
            # self.x_train, self.x_test = normalise_signals(self.x_train, self.x_test)

            if lstm_sequences:  # ======================================================================================
                # GROUP INTO SEQUENCES ---------------------------------------------------------------------------------
                self.x_train, self.y_train = group_windows_into_sequences(self.x_train, self.y_train,
                                                                          self.sequence_length, window_axis=-1)
                self.x_test, self.y_test = group_windows_into_sequences(self.x_test, self.y_test,
                                                                        self.sequence_length, window_axis=-1)
                # windowed signals of shape (n_windows_per_sequence, window_length, n_channels, n_sequences)
                # windowed labels of shape (n_windows_per_sequence, n_channels, n_sequences)

                # TRANSPOSE --------------------------------------------------------------------------------------------
                self.x_train = self.x_train.transpose((0, 2, 1, 3))
                self.x_test = self.x_test.transpose((0, 2, 1, 3))
                # windowed signals of shape (n_windows_per_sequence, n_channels, window_length, n_sequences)
                # windowed labels of shape (n_windows_per_sequence, n_channels, n_sequences)

            # SHUFFLE ALONG THE N_SEQUENCE AXIS ------------------------------------------------------------------------
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train)

            # BATCH THE TRAINING DATA ----------------------------------------------------------------------------------
            self.x_train, self.y_train, self.x_test, self.y_test = split_into_batches(self.x_train, self.y_train,
                                                                                      self.x_test, self.y_test,
                                                                                      self.batch_size, batch_axis=0)

            # x train of shape (sequence_length, batch_size, n_channels, window_length, n_batches)
            # y train of shape (sequence_length, batch_size, n_channels, n_batches)
            # x test of shape (sequence_length, 1, n_channels, window_length, n_testing_sequences)
            # y test of shape (sequence_length, 1, 1, n_testing_sequences)
            self.turn_into_tensors()

            # PRODUCE A FINAL ATTRIBUTE WITH ALL THE RELEVANT INFORMATION TO BE EASILY EXTRACTED ----------------------
            self.prepped_data = self.x_train, self.y_train, self.x_test, self.y_test

        else:
            average_values = []
            std_values = []
            for channel in range(self.EMG_signals.shape[0]):
                average_values.append(np.mean(self.EMG_signals[channel, :, :]))
                std_values.append(np.std(self.EMG_signals[channel, :, :], dtype=np.float64))
            self.norm_values = average_values, std_values
            # WINDOW THE SIGNAL ----------------------------------------------------------------------------------------
            self.windowed_signals, self.windowed_labels = split_signals_into_TCN_windows\
                (EMG_signals, labels, self.window_length, self.window_step, self.prediction_delay, False)
            # windowed signals of shape (window_length, n_channels, n_reps)
            # windowed labels of shape (n_channels, n_reps)

            if lstm_sequences:  # ======================================================================================
                # GROUP INTO SEQUENCES ---------------------------------------------------------------------------------
                self.windowed_signals, self.windowed_labels = group_windows_into_sequences\
                    (self.windowed_signals, self.windowed_labels, self.sequence_length, window_axis=-1)
                # windowed signals of shape (n_windows_per_sequence, window_length, n_channels, n_sequences)
                # windowed labels of shape (n_windows_per_sequence, n_channels, n_sequences)

                # TRANSPOSE --------------------------------------------------------------------------------------------
                self.windowed_signals = self.windowed_signals.transpose((0, 2, 1, 3))
                # windowed signals of shape (n_windows_per_sequence, n_channels, window_length, n_sequences)
                # windowed labels of shape (n_windows_per_sequence, n_channels, n_sequences)

            # TURN INTO TENSORS ----------------------------------------------------------------------------------------
            # self.turn_into_tensors()
            self.windowed_signals = torch.autograd.Variable(torch.from_numpy(self.windowed_signals), requires_grad=False)
            self.windowed_labels = torch.autograd.Variable(torch.from_numpy(self.windowed_labels), requires_grad=False)

            self.prepped_windowed_data = self.windowed_signals, self.windowed_labels

    def turn_into_tensors(self):
        self.x_train = torch.autograd.Variable(torch.from_numpy(self.x_train), requires_grad=False)
        self.y_train = torch.autograd.Variable(torch.from_numpy(self.y_train), requires_grad=False)
        self.x_test = torch.from_numpy(self.x_test)
        self.y_test = torch.from_numpy(self.y_test)


if __name__ == '__main__':
    hdemg_data, hdemg_labels = get_hdemg_data('BS03', 'Ham', 'Fast1')
    delsys_data, delsys_labels = get_delsys_data('Ham')
    hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels, delsys_data,
                                                                                  delsys_labels)
    plt.plot(hdemg_labels)
    plt.plot(delsys_labels)
    plt.show()
    source_data = list(TCNDataPrep(hdemg_data, hdemg_labels, window_length=512, window_step=40, batch_size=16,
                                   sequence_length=15, label_delay=0, training_size=0.8, lstm_sequences=False,
                                   split_data=True, shuffle_full_dataset=False).prepped_data)

    target_data = list(TCNDataPrep(delsys_data, delsys_labels, window_length=512, window_step=40, batch_size=16,
                                   sequence_length=15, label_delay=0, training_size=0.8, lstm_sequences=False,
                                   split_data=True, shuffle_full_dataset=False).prepped_data)

    print("For the source data we have:")
    print(source_data[0].shape, source_data[1].shape, source_data[2].shape, source_data[3].shape)
    print("For the target data we have:")
    print(target_data[0].shape, target_data[1].shape, target_data[2].shape, target_data[3].shape)




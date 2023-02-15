import math
import statistics
import scipy
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Preprocessing import get_delsys_data, get_hdemg_data, TCNDataPrep, align_signals_together
from utility.conversions import envelope
from plots import visualise_the_weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_signal_envelopes_comparison(average_cycle=True):
    list_of_muscles = ["Quad", "Ham", "Tibialis", "Soleus"]
    list_of_names = ['RF', 'BF', 'TA', 'SO']
    my_cmap = sns.color_palette("hls", 4).as_hex()[2::]
    fig = plt.figure(figsize=(6, 4.5))
    for i in range(len(list_of_muscles)):
        hdemg_data, hdemg_labels = get_hdemg_data('BS03', list_of_muscles[i], 'Fast1')
        delsys_data, delsys_labels = get_delsys_data(list_of_muscles[i])
        hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels,
                                                                                      delsys_data,
                                                                                      delsys_labels)
        hdemg_data = envelope(hdemg_data, fs=2000, lop=2)
        delsys_data = envelope(delsys_data, fs=2000, lop=2)
        if average_cycle:
            hdemg_label_peaks, _= find_peaks(hdemg_labels.squeeze(), height=0.6, distance=2000)
            hdemg_label_peaks = np.insert(hdemg_label_peaks, 0, 0)
            hdemg_label_peaks = np.append(hdemg_label_peaks, len(hdemg_labels))
            delsys_label_peaks, _ = find_peaks(delsys_labels.squeeze(), height=0.6, distance=2000)
            delsys_label_peaks = np.insert(delsys_label_peaks, 0, 0)
            delsys_label_peaks = np.append(delsys_label_peaks, len(delsys_labels))
            cycles = []
            for peak in range(1, len(hdemg_label_peaks)):
                cycle = hdemg_data[hdemg_label_peaks[peak-1]:hdemg_label_peaks[peak]]
                old_x = np.linspace(0, 1000, len(cycle))
                new_x = np.linspace(0, 1000, 100)
                f = interp1d(old_x, cycle.squeeze())
                cycle = f(new_x)
                cycles.append(cycle)
            average_hdemg_cycle= np.mean(cycles, axis=0)
            hdemg_cycle_std = np.std(cycles, axis=0)
            cycles = []
            for peak in range(1, len(delsys_label_peaks)):
                cycle = delsys_data[delsys_label_peaks[peak - 1]:delsys_label_peaks[peak]]
                old_x = np.linspace(0, 1000, len(cycle))
                new_x = np.linspace(0, 1000, 100)
                f = interp1d(old_x, cycle.squeeze())
                cycle = f(new_x)
                cycles.append(cycle)
            average_delsys_cycle = np.mean(cycles, axis=0)
            delsys_cycle_std = np.std(cycles, axis=0)
            print("For the muscle ", list_of_names[i])
            print("The average difference between the EMG signals is of ", np.mean(average_hdemg_cycle-average_delsys_cycle))
            print("The average HDEMG std is  ", np.mean(hdemg_cycle_std))
            print("The average Delsys std is ", np.mean(delsys_cycle_std))
            print("The average std difference is ", np.mean(hdemg_cycle_std-delsys_cycle_std))
            plt.subplot(2, 2, i+1)
            x = np.linspace(0, 100, len(average_hdemg_cycle))
            if i < 3:
                plt.plot(average_hdemg_cycle, color=my_cmap[0])
                plt.fill_between(x, average_hdemg_cycle-hdemg_cycle_std, average_hdemg_cycle+hdemg_cycle_std,
                                 color=my_cmap[0], alpha=0.2)
                plt.plot(average_delsys_cycle, color=my_cmap[1])
                plt.fill_between(x, average_delsys_cycle - delsys_cycle_std, average_delsys_cycle + delsys_cycle_std,
                                 color=my_cmap[1], alpha=0.2)
            else:
                plt.plot(average_hdemg_cycle, color=my_cmap[0], label='HDEMG')
                plt.fill_between(x, average_hdemg_cycle - hdemg_cycle_std, average_hdemg_cycle + hdemg_cycle_std,
                                 color=my_cmap[0], alpha=0.2)
                plt.plot(average_delsys_cycle, color=my_cmap[1], label='Delsys')
                plt.fill_between(x, average_delsys_cycle - delsys_cycle_std, average_delsys_cycle + delsys_cycle_std,
                                 color=my_cmap[1], alpha=0.2)
                plt.legend(ncol=2, bbox_to_anchor=(1, -0.35))
            maximum_value = np.max([average_hdemg_cycle + hdemg_cycle_std, average_delsys_cycle + delsys_cycle_std])
            minimum_value = np.min([average_hdemg_cycle - hdemg_cycle_std, average_delsys_cycle - delsys_cycle_std])
            plt.title(list_of_names[i])
            plt.xlabel('GC%')
            plt.ylabel('mV', loc='top', rotation=0, labelpad=-25)
            plt.ylim([minimum_value-0.009, maximum_value+0.009])
        else:
            plt.subplot(4, 1, i+1)
            if i < 3:
                plt.plot(hdemg_data[1000:30000, 0], color=my_cmap[0])
                plt.plot(delsys_data[1000:30000, 0], color=my_cmap[1])
            else:
                plt.plot(hdemg_data[1000:30000, 0], color=my_cmap[0], label='HDEMG')
                plt.plot(delsys_data[1000:30000, 0], color=my_cmap[1], label='Delsys')
                plt.legend(ncol=2, bbox_to_anchor=(0.7, -0.25))
            plt.title(list_of_names[i])
    fig.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig('/media/ag6016/Storage/DomainAdaptation/Images/muscle_envelopes.pdf',
                dpi=200, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(5)
    plt.close()



# TEMPORAL CONVOLUTIONAL NETWORK =======================================================================================
class TempConvNetwork(nn.Module):
    def __init__(self, n_inputs=1, kernel_size=5, stride=1, dilation=5, dropout=0.2, n_AE_layers=3):
        super(TempConvNetwork, self).__init__()
        # Here, define each layer with their inputs, for example:
        self.input_size = n_inputs
        self.flattened_length = 960
        self.AE_layers = n_AE_layers
        self.conv1 = nn.Conv1d(in_channels=self.input_size, out_channels=8, kernel_size=kernel_size, stride=stride,
                               dilation=dilation, padding='same')
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding='same')
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding='same')
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding='same')
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding='same')
        self.act = nn.LeakyReLU()
        self.pooling1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pooling2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pooling3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pooling4 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pooling5 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.TCN_trainer = nn.Sequential(
            # nn.BatchNorm1d(self.input_size),
            nn.Flatten(),
            nn.Linear(int(self.flattened_length), int(self.flattened_length / 2)),  # (1, 512)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 2), int(self.flattened_length / 4)),  # (1, 256)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 4), int(self.flattened_length / 8)),  # (1, 128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 8), int(self.flattened_length / 16)),  # (1, 128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 16), 1)
            )

    def forward(self, EMG_signal):
        # (batch, 1, 2048)
        outputs = []
        out = self.pooling1(self.act(self.conv1(EMG_signal)))
        outputs.append(out)
        out = self.pooling2(self.act(self.conv2(out)))
        outputs.append(out)
        out = self.pooling3(self.act(self.conv3(out)))
        outputs.append(out)
        out = self.pooling4(self.act(self.conv4(out)))
        outputs.append(out)
        out = self.pooling5(self.act(self.conv5(out)))
        outputs.append(out)
        out = self.TCN_trainer(out)
        extracted_features = nn.Flatten()(outputs[self.AE_layers])
        return extracted_features, out


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return(torch.mean(y_true * y_pred))


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)


class RunTCN:
    def __init__(self, n_channels, epochs, saved_model_name, load_model=False,
                 initial_lr=0.001, lr_update_step=3):

        self.model = TempConvNetwork(n_inputs=n_channels, kernel_size=5, stride=1, dilation=4, dropout=0.4).to(device)
        self.model_type = 'TCN'
        self.saved_model_name = saved_model_name
        self.saved_model_path = '/media/ag6016/Storage/MuscleSelection/Models/' + self.saved_model_name + '.pth'
        initialize_weights(self.model)
        if load_model:
            self.model.load_state_dict(torch.load(self.saved_model_path))
        self.criterion = nn.MSELoss().to(device)
        self.epochs = epochs
        self.writer = SummaryWriter()
        self.lr_update_step = lr_update_step
        self.initial_lr = initial_lr
        self.updated_lr = None
        self.recorded_training_error = 100
        self.recorded_validation_error = 100
        self.recorded_testing_error = None
        self.epochs_ran = 0
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None


    def train_network(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        rep_step = 0
        lowest_error = 1000.0
        cut_off_counter = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, betas=(0.9, 0.999))
        lr = self.initial_lr
        for epoch in range(self.epochs):
            print("Epoch number:", epoch)
            running_training_loss = 0.0
            running_validation_loss = 0.0
            for rep in tqdm(np.arange(self.x_train.shape[-1])):
                x_train = self.x_train[:, :, :, rep].to(device)
                y_train = self.y_train[:, :, rep].to(device)
                _, predicted = self.model.forward(EMG_signal=x_train.float())
                loss = self.criterion(predicted, y_train.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_training_loss += loss.item()
                rep_step += 1
            recorded_training_error = math.sqrt(running_training_loss / (self.x_train.shape[-1]))
            self.writer.add_scalar("Epoch training loss ", recorded_training_error, global_step=epoch)
            # VALIDATION LOOP
            with torch.no_grad():
                for rep in range(self.x_test.shape[-1]):
                    x_test = self.x_test[:, :, :, rep].to(device)
                    y_test = self.y_test[:, :, rep].to(device)
                    _, predicted = self.model.forward(EMG_signal=x_test.float())
                    validation_loss = self.criterion(predicted, y_test.float())
                    running_validation_loss += validation_loss.item()
            recorded_validation_error = math.sqrt(running_validation_loss / (self.x_test.shape[-1]))
            self.writer.add_scalar("Epoch val loss ", recorded_validation_error, global_step=epoch)
            if recorded_validation_error < lowest_error:
                torch.save(self.model.state_dict(), self.saved_model_path)
                lowest_error = recorded_validation_error
                self.recorded_validation_error = recorded_validation_error
                self.recorded_training_error = recorded_training_error
                print("The errors are ", self.recorded_training_error, self.recorded_validation_error)
                self.epochs_ran = epoch
                cut_off_counter = 0
                print("it's lower")
            else:
                cut_off_counter += 1

            if cut_off_counter > self.lr_update_step and lr == self.initial_lr:
                self.updated_lr = self.initial_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr == self.initial_lr / 10:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr == self.initial_lr / 100:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr == self.initial_lr / 1000:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr == self.initial_lr / 10000:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > 6:
                break

    def test_network(self, x_test_data, y_test_data):
        running_loss = 0.0
        with torch.no_grad():
            for rep in range(self.x_test.shape[-1]):
                x_test = x_test_data[:, :, :, rep].to(device)
                y_test = y_test_data[:, :, rep].to(device)
                _, predicted = self.model.forward(EMG_signal=x_test.float())
                loss = self.criterion(predicted, y_test.float())
                running_loss += loss.item()
        recorded_error = math.sqrt(running_loss / (self.x_test.shape[-1]))
        self.recorded_testing_error = recorded_error


class RunDomainAdaptationTCN:
    def __init__(self, n_channels, epochs, saved_model_name, AE_layers=3, load_model=False,
                 initial_lr=0.0001, lr_update_step=3, distribution_loss_ratio=0.7, fix_weights=False, loss='sinkhorn'):
        self.model = TempConvNetwork(n_inputs=n_channels, kernel_size=5, stride=1, dilation=4, dropout=0.4).to(device)
        self.model_type = 'TCN'
        self.saved_model_name = saved_model_name
        self.saved_model_path = '/media/ag6016/Storage/DomainAdaptation/Models/' + self.saved_model_name + '.pth'
        self.initialise_weights()
        if load_model:
            self.model.load_state_dict(torch.load(self.saved_model_path))
        else:
            self.model.apply(initialize_weights)
        if fix_weights:
            self.fix_weights()
        self.criterion = nn.MSELoss().to(device)

        if loss == 'sinkhorn':
            self.distribution_loss = SinkhornDistance(eps=0.1, max_iter=100).to(device)
        elif loss == 'mse':
            self.distribution_loss = nn.MSELoss().to(device)
        elif loss == 'wasserstein':
            self.distribution_loss = WassersteinLoss().to(device)
        else:
            raise ValueError('Please choose either sinkhorn, mse, or wasserstein')

        self.distribution_loss = SinkhornDistance(eps=0.1, max_iter=100).to(device)#nn.MSELoss().to(device)
        self.epochs = epochs
        self.writer = SummaryWriter()
        self.lr_update_step = lr_update_step
        self.initial_lr = initial_lr
        self.loss_ratio = distribution_loss_ratio
        self.updated_lr = None
        self.AE_layers = AE_layers
        self.recorded_training_error = 100
        self.recorded_validation_error = 100
        self.recorded_validation_error_source = None
        self.recorded_validation_error_target = None
        self.epochs_ran = 0
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.target_x_train = self.target_y_train = self.target_x_test = self.target_y_test = None

    def fix_weights(self):
        if self.AE_layers == 3 or self.AE_layers>3:
            for p in self.model.conv1.parameters():
                p.requires_grad = False
            for p in self.model.pooling1.parameters():
                p.requires_grad = False
            for p in self.model.conv2.parameters():
                p.requires_grad = False
            for p in self.model.pooling2.parameters():
                p.requires_grad = False
            for p in self.model.conv3.parameters():
                p.requires_grad = False
            for p in self.model.pooling3.parameters():
                p.requires_grad = False
        if self.AE_layers == 4 or self.AE_layers>4:
            for p in self.model.conv4.parameters():
                p.requires_grad = False
            for p in self.model.pooling4.parameters():
                p.requires_grad = False
        if self.AE_layers == 5:
            for p in self.model.conv5.parameters():
                p.requires_grad = False
            for p in self.model.pooling5.parameters():
                p.requires_grad = False

    def initialise_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def initialise_remaining_weights(self):
        fix_layer_counter = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                fix_layer_counter += 1
                if fix_layer_counter > self.AE_layers:
                    nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)

    def train_network(self, source_data, target_data):
        self.x_train, self.y_train, self.x_test, self.y_test = source_data
        self.target_x_train, self.target_y_train, self.target_x_test, self.target_y_test = target_data
        shuffle_idx = torch.randperm(self.x_train.shape[-1])
        self.x_train = self.x_train[:, :, :, shuffle_idx]
        self.y_train = self.y_train[:, :, shuffle_idx]
        self.target_x_train = self.target_x_train[:, :, :, shuffle_idx]
        self.target_y_train = self.target_y_train[:, :, shuffle_idx]
        rep_step = 0
        lowest_error = 1000.0
        cut_off_counter = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, betas=(0.9, 0.999))
        lr = self.initial_lr
        for epoch in range(self.epochs):
            print("Epoch number:", epoch)
            running_pred_train_loss = 0.0
            running_AE_loss = 0.0
            running_training_loss = 0.0
            running_validation_loss_source = 0.0
            running_validation_loss_target = 0.0
            for rep in tqdm(np.arange(self.x_train.shape[-1])):
                x_train = self.x_train[:, :, :, rep].to(device)
                y_train = self.y_train[:, :, rep].to(device)
                source_features, predicted = self.model.forward(EMG_signal=x_train.float())
                target_x_train = self.target_x_train[:, :, :, rep].to(device)
                target_features, _ = self.model.forward(EMG_signal=target_x_train.float())
                distribution_loss, P, C = self.distribution_loss(source_features, target_features)
                #distribution_loss = wasserstein_loss(source_features, target_features)

                running_AE_loss += distribution_loss.item()
                prediction_loss = self.criterion(predicted, y_train.float())
                running_pred_train_loss += prediction_loss.item()
                loss = self.loss_ratio*distribution_loss + (1-self.loss_ratio)*prediction_loss
                running_training_loss += loss.item()
                # self.writer.add_scalar("Total training loss ", loss.item(), global_step=rep_step)
                # self.writer.add_scalar("Prediction loss ", prediction_loss.item(), global_step=rep_step)
                # self.writer.add_scalar("AE loss ", distribution_loss.item(), global_step=rep_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rep_step += 1
            recorded_training_error = math.sqrt(abs(running_training_loss) / (self.x_train.shape[-1]))
            recorded_AE_error = math.sqrt(abs(running_AE_loss) / (self.x_train.shape[-1]))
            recorded_pred_error = math.sqrt(abs(running_pred_train_loss) / (self.x_train.shape[-1]))
            self.writer.add_scalar("Total training loss ", recorded_training_error, global_step=epoch)
            self.writer.add_scalar("AE training loss ", recorded_AE_error, global_step=epoch)
            self.writer.add_scalar("Prediction training loss ", recorded_pred_error, global_step=epoch)
            # VALIDATION LOOP
            with torch.no_grad():
                for rep in range(self.x_test.shape[-1]):
                    x_test = self.x_test[:, :, :, rep].to(device)
                    y_test = self.y_test[:, :, rep].to(device)
                    _, predicted = self.model.forward(EMG_signal=x_test.float())
                    validation_loss = self.criterion(predicted, y_test.float())
                    running_validation_loss_source += validation_loss.item()
            recorded_validation_error_source = math.sqrt(abs(running_validation_loss_source) / (self.x_test.shape[-1]))
            self.recorded_validation_error_source = recorded_validation_error_source
            self.writer.add_scalar("Epoch source val loss ", recorded_validation_error_source, global_step=epoch)
            with torch.no_grad():
                for rep in range(self.target_x_test.shape[-1]):
                    x_test = self.target_x_test[:, :, :, rep].to(device)
                    y_test = self.target_y_test[:, :, rep].to(device)
                    _, predicted = self.model.forward(EMG_signal=x_test.float())
                    validation_loss = self.criterion(predicted, y_test.float())
                    running_validation_loss_target += validation_loss.item()
            recorded_validation_error_target = math.sqrt(abs(running_validation_loss_target) / (self.x_test.shape[-1]))
            self.recorded_validation_error_target = recorded_validation_error_target
            self.writer.add_scalar("Epoch target val loss ", recorded_validation_error_target, global_step=epoch)
            recorded_validation_error = recorded_validation_error_source + recorded_validation_error_target
            if recorded_validation_error < lowest_error:
                torch.save(self.model.state_dict(), self.saved_model_path)
                lowest_error = recorded_validation_error
                self.recorded_validation_error = recorded_validation_error
                self.recorded_training_error = recorded_training_error
                print("The errors are ", self.recorded_training_error, self.recorded_validation_error)
                self.epochs_ran = epoch
                cut_off_counter = 0
                print("it's lower")
            else:
                cut_off_counter += 1

            if cut_off_counter > self.lr_update_step and lr == self.initial_lr:
                self.updated_lr = self.initial_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr == self.initial_lr / 10:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr ==self.initial_lr / 100:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr == self.initial_lr / 1000:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > self.lr_update_step and lr == self.initial_lr / 10000:
                self.updated_lr = self.updated_lr / 10
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.updated_lr, betas=(0.9, 0.999))
                self.model.load_state_dict(torch.load(self.saved_model_path))
                cut_off_counter = 0
                lr = self.updated_lr
                print("update of the learning rate to ", lr)
            elif cut_off_counter > 6:
                break

    def test_network(self, x_test_data, y_test_data):
        running_loss = 0.0
        with torch.no_grad():
            for rep in range(self.x_test.shape[-1]):
                x_test = x_test_data[:, :, :, rep].to(device)
                y_test = y_test_data[:, :, rep].to(device)
                _, predicted = self.model.forward(EMG_signal=x_test.float())
                loss = self.criterion(predicted, y_test.float())
                running_loss += loss.item()
        recorded_error = math.sqrt(running_loss / (self.x_test.shape[-1]))
        return recorded_error


class CheckDAUsefulness:
    def __init__(self, subject, muscle, original_model_name, DA_model_name, epochs=100, original_model_lr=0.0001,
                 DA_model_lr=0.00001, lr_update_step=4, AE_layers=4, distribution_loss_ratio=0.8):
        self.subject = subject  # As a string
        self.muscle = muscle  # As a string
        self.original_model = RunTCN(n_channels=1, epochs=epochs, saved_model_name=self.muscle+original_model_name,
                                     initial_lr=original_model_lr, lr_update_step=lr_update_step)
        self.DA_model_mse = RunDomainAdaptationTCN(n_channels=1, epochs=epochs,
                                                           saved_model_name=muscle + DA_model_name,
                                                           AE_layers=AE_layers, initial_lr=DA_model_lr,
                                                           distribution_loss_ratio=distribution_loss_ratio,
                                                           lr_update_step=lr_update_step, loss='mse')
        self.DA_model_sinkhorn = RunDomainAdaptationTCN(n_channels=1, epochs=epochs,
                                                        saved_model_name=muscle+DA_model_name,
                                                        AE_layers=AE_layers, initial_lr=DA_model_lr,
                                                        distribution_loss_ratio=distribution_loss_ratio,
                                                        lr_update_step=lr_update_step, loss='sinkhorn')
        self.DA_model_wasserstein = RunDomainAdaptationTCN(n_channels=1, epochs=epochs,
                                                           saved_model_name=muscle+DA_model_name,
                                                           AE_layers=AE_layers, initial_lr=DA_model_lr,
                                                           distribution_loss_ratio=distribution_loss_ratio,
                                                           lr_update_step=lr_update_step, loss='wasserstein')
        self.source_data = None  # This will be a list of the x_train, y_train, x_test and y_test values
        self.target_data = None  # This will be a list of the x_train, y_train, x_test and y_test values
        self.original_vs_DA_acc_report = None
        self.get_data()

    def get_data(self):
        """
        This extracts the correct data for both the HDEMG and the Delsys and creates the source and target datasets
        :return: updates the self.source_data and self.target_data values, which are lists containing x_train, y_train,
        x_test, and y_test
        """
        hdemg_data, hdemg_labels = get_hdemg_data(self.subject, self.muscle, 'Fast1')
        delsys_data, delsys_labels = get_delsys_data(self.muscle)
        hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels,
                                                                                      delsys_data,
                                                                                      delsys_labels)
        self.source_data = list(TCNDataPrep(hdemg_data, hdemg_labels, window_length=512, window_step=40, batch_size=16,
                                       sequence_length=15, label_delay=0, training_size=0.8, lstm_sequences=False,
                                       split_data=True, shuffle_full_dataset=False).prepped_data)

        self.target_data = list(TCNDataPrep(delsys_data, delsys_labels, window_length=512, window_step=40, batch_size=16,
                                       sequence_length=15, label_delay=0, training_size=0.8, lstm_sequences=False,
                                       split_data=True, shuffle_full_dataset=False).prepped_data)

    def compare_original_vs_DA_model(self, index_name):
        """
        This is a function that will determine the difference in performance between the original and the DA models
        :return: A Dataframe of the validation accuracies for comparison
        """

        self.original_model.train_network(self.source_data[0], self.source_data[1], self.source_data[2],
                                          self.source_data[3])
        original_hdemg_val_acc = 1 - (self.original_model.recorded_validation_error /
                                      (torch.max(self.source_data[3]).item() - torch.min(self.source_data[3]).item()))
        self.original_model.test_network(self.target_data[2], self.target_data[3])
        original_delsys_val_acc = 1 - (self.original_model.recorded_testing_error /
                                      (torch.max(self.target_data[3]).item() - torch.min(self.target_data[3]).item()))

        self.DA_model_mse.train_network(self.source_data, self.target_data)
        DA_hdemg_val_acc_mse = 1 - (self.DA_model_mse.recorded_validation_error_source /
                                         (torch.max(self.source_data[3]).item() - torch.min(
                                             self.source_data[3]).item()))
        DA_delsys_val_acc_mse = 1 - (self.DA_model_mse.recorded_validation_error_target /
                                          (torch.max(self.source_data[3]).item() - torch.min(
                                              self.source_data[3]).item()))

        self.DA_model_sinkhorn.train_network(self.source_data, self.target_data)
        DA_hdemg_val_acc_sinkhorn = 1 - (self.DA_model_sinkhorn.recorded_validation_error_source /
                                      (torch.max(self.source_data[3]).item() - torch.min(self.source_data[3]).item()))
        DA_delsys_val_acc_sinkhorn = 1 - (self.DA_model_sinkhorn.recorded_validation_error_target /
                                      (torch.max(self.source_data[3]).item() - torch.min(self.source_data[3]).item()))

        self.DA_model_wasserstein.train_network(self.source_data, self.target_data)
        DA_hdemg_val_acc_wasserstein = 1 - (self.DA_model_wasserstein.recorded_validation_error_source /
                                         (torch.max(self.source_data[3]).item() - torch.min(
                                             self.source_data[3]).item()))
        DA_delsys_val_acc_wasserstein = 1 - (self.DA_model_wasserstein.recorded_validation_error_target /
                                          (torch.max(self.source_data[3]).item() - torch.min(
                                              self.source_data[3]).item()))

        self.original_vs_DA_acc_report = pd.DataFrame([[original_hdemg_val_acc, original_delsys_val_acc,
                                                        DA_hdemg_val_acc_mse, DA_delsys_val_acc_mse,
                                                        DA_hdemg_val_acc_sinkhorn,  DA_delsys_val_acc_sinkhorn,
                                                        DA_hdemg_val_acc_wasserstein,  DA_delsys_val_acc_wasserstein]],
                                                      columns=['Tested on HDEMG', 'Tested on Delsys',
                                                               'Tested on HDEMG with DA MSE',
                                                               'Tested on Delsys with DA MSE',
                                                               'Tested on HDEMG with DA Sinkhorn',
                                                               'Tested on Delsys with DA Sinkhorn',
                                                               'Tested on HDEMG with DA Wasserstein',
                                                               'Tested on Delsys with DA Wasserstein'],
                                                      index=[index_name])


class CheckDAGeneralisation:
    """
    This class is to try and see how well the DA generalises across muscle groups
    """
    def __init__(self, muscles,  original_model_name, DA_model_name, epochs=100, original_model_lr=0.0001,
                 DA_model_lr=0.00001, lr_update_step=4, AE_layers=4, distribution_loss_ratio=0.8,
                 across_subjects=False, across_muscles=False):
        self.across_subjects = across_subjects
        self.across_muscles = across_muscles
        if self.across_muscles:
            self.training_muscle = muscles[0]  # Assume that the muscle group being trained for the DA is the first one
            self.testing_muscle = muscles[1]  # The muscle group where the DA is applied and tested is the second
        else:
            self.training_muscle=muscles
        self.original_model = RunTCN(n_channels=1, epochs=epochs,
                                     saved_model_name=original_model_name,
                                     initial_lr=original_model_lr, lr_update_step=lr_update_step)
        self.DA_model_mse = RunDomainAdaptationTCN(n_channels=1, epochs=epochs,
                                                        saved_model_name='MSE'+DA_model_name,
                                                        AE_layers=AE_layers, initial_lr=DA_model_lr,
                                                        distribution_loss_ratio=distribution_loss_ratio,
                                                        lr_update_step=lr_update_step, loss='mse')
        self.DA_model_sinkhorn = RunDomainAdaptationTCN(n_channels=1, epochs=epochs,
                                               saved_model_name='Sinkhorn'+DA_model_name,
                                               AE_layers=AE_layers, initial_lr=DA_model_lr,
                                               distribution_loss_ratio=distribution_loss_ratio,
                                               lr_update_step=lr_update_step, loss='sinkhorn')
        self.DA_model_wasserstein = RunDomainAdaptationTCN(n_channels=1, epochs=epochs,
                                                        saved_model_name='Wasserstein'+DA_model_name,
                                                        AE_layers=AE_layers, initial_lr=DA_model_lr,
                                                        distribution_loss_ratio=distribution_loss_ratio,
                                                        lr_update_step=lr_update_step, loss='wasserstein')
        self.training_subject = 'BS03'
        if self.across_subjects:
            self.testing_subject = None
        self.training_source_data = None  # This will be a list of the x_train, y_train, x_test and y_test values
        self.training_target_data = None  # This will be a list of the x_train, y_train, x_test and y_test values
        self.testing_source_data = None  # This will be a list of the x_train, y_train, x_test and y_test values
        self.testing_target_data = None  # This will be a list of the x_train, y_train, x_test and y_test values
        self.DA_generalisation_report = None
        self.get_data()


    def get_data(self):
        """
        This extracts the correct data for both the HDEMG and the Delsys and creates the source and target datasets
        :return: updates the self.source_data and self.target_data values, which are lists containing x_train, y_train,
        x_test, and y_test
        """
        hdemg_data, hdemg_labels = get_hdemg_data(self.training_subject, self.training_muscle, 'Fast1')
        delsys_data, delsys_labels = get_delsys_data(self.training_muscle)
        hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels,
                                                                                      delsys_data,
                                                                                      delsys_labels)
        # Recall that the training_source_data corresponds to the muscle used to train the DA, so first muscle
        self.training_source_data = list(TCNDataPrep(hdemg_data, hdemg_labels, window_length=512, window_step=40,
                                                     batch_size=16,sequence_length=15, label_delay=0, training_size=0.8,
                                                     lstm_sequences=False, split_data=True,
                                                     shuffle_full_dataset=False).prepped_data)
        # The training_target_data is the Delsys signal used to train the DA, so also the first muscle
        self.training_target_data = list(TCNDataPrep(delsys_data, delsys_labels, window_length=512, window_step=40,
                                                     batch_size=16, sequence_length=15, label_delay=0,
                                                     training_size=0.8, lstm_sequences=False, split_data=True,
                                                     shuffle_full_dataset=False).prepped_data)
        if self.across_muscles:
            print("We have entered the loop for the analysis across muscles")
            hdemg_data, hdemg_labels = get_hdemg_data(self.training_subject, self.testing_muscle, 'Fast1')
            delsys_data, delsys_labels = get_delsys_data(self.testing_muscle)
            hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels,
                                                                                          delsys_data,
                                                                                          delsys_labels)
            # Now the testing source data is for the HDEMG signal from the source muscle
            self.testing_source_data = list(TCNDataPrep(hdemg_data, hdemg_labels, window_length=512, window_step=40,
                                                         batch_size=16, sequence_length=15, label_delay=0,
                                                         training_size=0.8,
                                                         lstm_sequences=False, split_data=True,
                                                         shuffle_full_dataset=False).prepped_data)
            # The testing target data is for the Delsys signal from the source muscle
            self.testing_target_data = list(TCNDataPrep(delsys_data, delsys_labels, window_length=512, window_step=40,
                                                         batch_size=16, sequence_length=15, label_delay=0,
                                                         training_size=0.8, lstm_sequences=False, split_data=True,
                                                         shuffle_full_dataset=False).prepped_data)

        elif self.across_subjects:
            pass

    def compare_DA_generalisation(self, index_name):
        """
        This function is to see whether the domain adaptation that we have can be trained on one muscle set and applied
        to another muscle, i.e. we want to know whether the way the DA model extracts data is the same regardless of
        the data type we are looking at.
        Note that this means that we will train the DA block on a completely different muscle group, and then fix
        those weights, and just train the model with the leftover layers that were not fixed from the DA block
        We can still rerun this for all the possible combinations of LRs and layers
        :return: A pd.Dataframe with the columns: Trained and tested on HDEMG, Tested on Delsys, Tested on HDEMG with DA
        trained on another model, Tested on Delsys with DA trained on another model
        """
        # Here we are training on the testing muscle group to get the original performance with trained on hdemg and
        # tested on delsys
        self.original_model.train_network(self.testing_source_data[0], self.testing_source_data[1], self.testing_source_data[2],
                                          self.testing_source_data[3])
        # visualise_the_weights(self.original_model.model)
        # plt.show()
        # exit()
        original_hdemg_val_acc = 1 - (self.original_model.recorded_validation_error /
                                      (torch.max(self.testing_source_data[3]).item() -
                                       torch.min(self.testing_source_data[3]).item()))
        self.original_model.test_network(self.testing_target_data[2], self.testing_target_data[3])
        original_delsys_val_acc = 1 - (self.original_model.recorded_testing_error /
                                       (torch.max(self.testing_target_data[3]).item() -
                                        torch.min(self.testing_target_data[3]).item()))
        print("Original model trained and tested")

        # Next we train a DA model with the training muscle group completely ===========================================
        self.DA_model_mse.train_network(self.training_source_data, self.training_target_data)

        # Now the DA model has been trained to fit the training muscle data. We need to fix those weights and update
        # the model
        self.DA_model_mse.fix_weights()
        self.DA_model_mse.initialise_remaining_weights()
        # Now we retrain the model with the testing muscle data
        self.DA_model_mse.train_network(self.testing_source_data, self.testing_target_data)

        DA_hdemg_val_acc_mse = 1 - (self.DA_model_mse.recorded_validation_error_source /
                                    (torch.max(self.testing_source_data[3]).item() - torch.min(
                                        self.testing_source_data[3]).item()))
        DA_delsys_val_acc_mse = 1 - (self.DA_model_mse.recorded_validation_error_target /
                                     (torch.max(self.testing_source_data[3]).item() - torch.min(
                                         self.testing_source_data[3]).item()))
        print("Trained with new ")
        # Now we can do the same thing with all the different losses
        # SINKHORN =====================================================================================================
        self.DA_model_sinkhorn.train_network(self.training_source_data, self.training_target_data)

        # Now the DA model has been trained to fit the training muscle data. We need to fix those weights and update
        # the model
        self.DA_model_sinkhorn.fix_weights()
        self.DA_model_sinkhorn.initialise_remaining_weights()

        # Now we retrain the model with the testing muscle data
        self.DA_model_sinkhorn.train_network(self.testing_source_data, self.testing_target_data)
        DA_hdemg_val_acc_sinkhorn = 1 - (self.DA_model_sinkhorn.recorded_validation_error_source /
                                    (torch.max(self.testing_source_data[3]).item() - torch.min(
                                        self.testing_source_data[3]).item()))
        DA_delsys_val_acc_sinkhorn = 1 - (self.DA_model_sinkhorn.recorded_validation_error_target /
                                     (torch.max(self.testing_source_data[3]).item() - torch.min(
                                         self.testing_source_data[3]).item()))

        # WASSERSTEIN =====================================================================================================
        self.DA_model_wasserstein.train_network(self.training_source_data, self.training_target_data)

        # Now the DA model has been trained to fit the training muscle data. We need to fix those weights and update
        # the model
        self.DA_model_wasserstein.fix_weights()
        self.DA_model_wasserstein.initialise_remaining_weights()

        # Now we retrain the model with the testing muscle data
        self.DA_model_wasserstein.train_network(self.testing_source_data, self.testing_target_data)
        DA_hdemg_val_acc_wasserstein = 1 - (self.DA_model_wasserstein.recorded_validation_error_source /
                                         (torch.max(self.testing_source_data[3]).item() - torch.min(
                                             self.testing_source_data[3]).item()))
        DA_delsys_val_acc_wasserstein = 1 - (self.DA_model_wasserstein.recorded_validation_error_target /
                                          (torch.max(self.testing_source_data[3]).item() - torch.min(
                                              self.testing_source_data[3]).item()))

        # Now we save everything into a nice Dataframe
        self.DA_generalisation_report = pd.DataFrame([[original_hdemg_val_acc, original_delsys_val_acc,
                                                                DA_hdemg_val_acc_mse, DA_delsys_val_acc_mse,
                                                                DA_hdemg_val_acc_sinkhorn,  DA_delsys_val_acc_sinkhorn,
                                                                DA_hdemg_val_acc_wasserstein,  DA_delsys_val_acc_wasserstein]],
                                                              columns=['Tested on HDEMG', 'Tested on Delsys',
                                                                       'Tested on HDEMG with DA MSE',
                                                                       'Tested on Delsys with DA MSE',
                                                                       'Tested on HDEMG with DA Sinkhorn',
                                                                       'Tested on Delsys with DA Sinkhorn',
                                                                       'Tested on HDEMG with DA Wasserstein',
                                                                       'Tested on Delsys with DA Wasserstein'],
                                                              index=[index_name])








































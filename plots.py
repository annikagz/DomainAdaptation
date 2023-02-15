import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def plot_model_performance(muscle):
    losses = ['MSE', 'Sinkhorn', 'Wasserstein']
    colours = ['#1F618D', '#5499C7', '#D4E6F1'] #['#154360', '#1F618D', '#2471A3', '#5499C7', '#7FB3D5', '#D4E6F1']
    DA_performance_report = pd.read_csv('/media/ag6016/Storage/DomainAdaptation/Reports/' + muscle + '_specific_performance.csv', index_col=0)
    index_columns = list(DA_performance_report.index)
    x_range = np.arange(len(index_columns))
    x_axis= ['3', '4', '5'] * 3
    plt.figure(figsize=(5.6, 6))
    plt.subplot(2, 1, 1)
    max_value = [max(DA_performance_report['Tested on HDEMG'])]
    min_value = [min(DA_performance_report['Tested on HDEMG'])]
    plt.plot(x_range, DA_performance_report['Tested on HDEMG'], '--k', label='No DA')
    for loss in range(len(losses)):
        plt.plot(x_range, DA_performance_report['Tested on HDEMG with DA '+losses[loss]], color=colours[loss], linestyle='-',
                 marker='o', markersize=5, label=losses[loss])
        max_value.append(max(DA_performance_report['Tested on HDEMG with DA '+losses[loss]]))
        min_value.append(min(DA_performance_report['Tested on HDEMG with DA '+losses[loss]]))
    max_value = max(max_value) + 0.02
    min_value = min(min_value) - 0.03
    plt.axvspan(-0.5, 2.5, alpha=0.03, color='black')
    plt.axvspan(2.51, 5.5, alpha=0.06, color='black')
    plt.axvspan(5.51, 8.5, alpha=0.09, color='black')
    plt.text(0.2, min_value+0.01, 'LR = 0.0001')
    plt.text(3.1, min_value+0.01, 'LR = 0.00001')
    plt.text(6, min_value+0.01, 'LR = 0.000001')
    plt.xlim([-0.5, 8.5])
    plt.ylim([min_value, max_value])
    plt.xticks(x_range, x_axis)
    plt.xlabel('Number of layers')
    plt.ylabel('Validation accuracy')
    plt.title('Tested on HDEMG')
    plt.subplot(2, 1, 2)
    max_value = [max(DA_performance_report['Tested on Delsys'])]
    min_value = [min(DA_performance_report['Tested on Delsys'])]
    plt.plot(x_range, DA_performance_report['Tested on Delsys'], '--k', label='No DA')
    for loss in range(len(losses)):
        plt.plot(x_range, DA_performance_report['Tested on Delsys with DA ' + losses[loss]], color=colours[loss], linestyle='-',
                 marker='o', markersize=5, label=losses[loss])
        max_value.append(max(DA_performance_report['Tested on Delsys with DA ' + losses[loss]]))
        min_value.append(min(DA_performance_report['Tested on Delsys with DA ' + losses[loss]]))
    max_value = max(max_value) + 0.02
    min_value = min(min_value) - 0.03
    plt.axvspan(-0.5, 2.5, alpha=0.03, color='black')
    plt.axvspan(2.51, 5.5, alpha=0.06, color='black')
    plt.axvspan(5.51, 8.5, alpha=0.09, color='black')
    plt.text(0.2, min_value + 0.01, 'LR = 0.0001')
    plt.text(3.1, min_value + 0.01, 'LR = 0.00001')
    plt.text(6, min_value + 0.01, 'LR = 0.000001')
    plt.xlim([-0.5, 8.5])
    plt.xticks(x_range, x_axis)
    plt.xlabel('Number of layers')
    plt.ylabel('Validation accuracy')
    plt.legend(ncol=4, bbox_to_anchor=(0.98, -0.25))
    plt.title('Tested on Delsys')
    plt.ylim([min_value, max_value])
    # plt.suptitle(muscle)
    plt.tight_layout()
    plt.savefig('/media/ag6016/Storage/DomainAdaptation/Reports/' + muscle + '_specific_performance.pdf',
                dpi=200, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def visualise_the_weights(model, linetype='-'):
    """
    We want to visualise the weights of the model to be able to compare the weights before and after the initialisation
    or the fixed weights
    :param model: the model that we are looking at that is either pre-trained or untrained that we will extract weights
    from
    :param linetype: for the graph plotting
    :return: a plot of the model's weights that will facilitate comparison
    """
    layer_number = 0
    colours = ['b', 'g', 'r', 'c', 'm', 'y']
    for m in model.modules():
        print(m)
        if isinstance(m, nn.Conv1d):
            weights = m.weight.data.cpu().numpy()
            plt.plot(weights[0, 0, :], color=colours[layer_number], linestyle=linetype, label='Layer '+ str(layer_number))
            layer_number += 1


if __name__ == '__main__':
    muscle_list = ['Quad', 'Ham', 'Tibialis', 'Soleus']
    for muscle in muscle_list:
        plot_model_performance(muscle)



import pandas as pd
from itertools import combinations
from Preprocessing import get_delsys_data, get_hdemg_data, TCNDataPrep, align_signals_together
from Networks import RunTCN, RunDomainAdaptationTCN, CheckDAUsefulness, CheckDAGeneralisation, plot_signal_envelopes_comparison

list_of_muscles = ["Quad", "Ham", "Tibialis", "Soleus"]

# MUSCLE SPECIFIC COMPARISON ===========================================================================================
# First, we are going to generate lots of differetn tables that are comparing the different performances of the DA model
# compared to the regular TCN model. We can do this for different number of layers and different learning rates
# We can experiment with 3 layers and 3 learning rates to begin with, and see from there
# We can also experiment with the loss function we are using for the Domain adaptation loss

plot_signal_envelopes_comparison()
exit()

list_of_lr = [0.0001, 0.00005, 0.00001]
list_of_layers = [3, 4, 5]
# list_of_losses = ['sinkhorn', 'wasserstein']
# for loss in list_of_losses:
muscle_combinations = []
for n in combinations(list_of_muscles, 2):
    muscle_combinations.append(list(n))

# for muscles in muscle_combinations:
#     muscle_and_subject_specific = pd.DataFrame(columns=['Tested on HDEMG', 'Tested on Delsys',
#                                                                'Tested on HDEMG with DA MSE',
#                                                                'Tested on Delsys with DA MSE',
#                                                                'Tested on HDEMG with DA Sinkhorn',
#                                                                'Tested on Delsys with DA Sinkhorn',
#                                                                'Tested on HDEMG with DA Wasserstein',
#                                                                'Tested on Delsys with DA Wasserstein'])
#     for lr in list_of_lr:
#         for n_layers in list_of_layers:
#             # We change the 0 and the 1 so we get the permutation of the muscles as well
#             model = CheckDAGeneralisation(muscles, original_model_name='_'+ muscles[1]+ '_' + muscles[0]+'_original',
#                                           DA_model_name='_'+ muscles[1]+muscles[0]+'_DA', epochs=100, DA_model_lr=lr,
#                                           AE_layers=n_layers, across_muscles=True)
#             model.compare_DA_generalisation(index_name=str(str(n_layers)+' layers with lr of '+str(lr)))
#             new_row = model.DA_generalisation_report
#             muscle_and_subject_specific = pd.concat([muscle_and_subject_specific, new_row])
#             muscle_and_subject_specific.to_csv('/media/ag6016/Storage/DomainAdaptation/Reports/' +
#                                                muscles[1]+ '_' + muscles[0] + '_generalisation_performance.csv')
#     muscle_and_subject_specific.to_csv('/media/ag6016/Storage/DomainAdaptation/Reports/' + muscles[1]+ '_' + muscles[0] +
#                                        '_generalisation_performance.csv')
#
# exit()

# for muscle in range(len(list_of_muscles)):
#     muscle_and_subject_specific = pd.DataFrame(columns=['Tested on HDEMG', 'Tested on Delsys',
#                                                                'Tested on HDEMG with DA MSE',
#                                                                'Tested on Delsys with DA MSE',
#                                                                'Tested on HDEMG with DA Sinkhorn',
#                                                                'Tested on Delsys with DA Sinkhorn',
#                                                                'Tested on HDEMG with DA Wasserstein',
#                                                                'Tested on Delsys with DA Wasserstein'])
#     for lr in list_of_lr:
#         for n_layers in list_of_layers:
#             model = CheckDAUsefulness(subject='BS03', muscle=list_of_muscles[muscle],
#                                       original_model_name=str(n_layers) + '_layers_' + str(lr),
#                                       DA_model_name=str(n_layers) + '_layers_DA_' + str(lr),
#                                       DA_model_lr=lr, AE_layers=n_layers, epochs=200)
    #         model.compare_original_vs_DA_model(index_name=str(str(n_layers)+' layers with lr of '+str(lr)))
    #         new_row = model.original_vs_DA_acc_report
    #         muscle_and_subject_specific = pd.concat([muscle_and_subject_specific, new_row])
    #         muscle_and_subject_specific.to_csv('/media/ag6016/Storage/DomainAdaptation/Reports/' +
    #                                            list_of_muscles[muscle] + '_specific_performance.csv')
    # muscle_and_subject_specific.to_csv('/media/ag6016/Storage/DomainAdaptation/Reports/' + list_of_muscles[muscle] +
    #                                    '_specific_performance.csv')

exit()






# for muscle_idx in range(len(list_of_muscles)):
#     new_row = []
#     hdemg_data, hdemg_labels = get_hdemg_data('BS03', list_of_muscles[muscle_idx], 'Fast1')
#     delsys_data, delsys_labels = get_delsys_data(list_of_muscles[muscle_idx])
#     hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels, delsys_data,
#                                                                                   delsys_labels)
#     source_data = list(TCNDataPrep(hdemg_data, hdemg_labels, window_length=512, window_step=40, batch_size=16,
#                                    sequence_length=15, label_delay=0, training_size=0.8, lstm_sequences=False,
#                                    split_data=True, shuffle_full_dataset=False).prepped_data)
#
#     target_data = list(TCNDataPrep(delsys_data, delsys_labels, window_length=512, window_step=40, batch_size=16,
#                                    sequence_length=15, label_delay=0, training_size=0.8, lstm_sequences=False,
#                                    split_data=True, shuffle_full_dataset=False).prepped_data)
#
#     # Calculate the performance when trained on HDEMG and tested on HDEMG
#     original_model = RunTCN(source_data[0], source_data[1], source_data[2], source_data[3], n_channels=1, epochs=100,
#                             saved_model_name=list_of_muscles[muscle_idx]+'_hdemg_trained', initial_lr=0.0001,
#                             lr_update_step=4)
#     original_model.train_network()
#     new_row.append(original_model.recorded_validation_error)
#
#     # Calculate the performance when trained on HDEMG and tested on Delsys without the DA
#     original_model.test_network(target_data[2], target_data[3])
#     new_row.append(original_model.recorded_testing_error)
#
#     DA_model = RunDomainAdaptationTCN(source_data, target_data, n_channels=1, epochs=100,
#                                       saved_model_name=list_of_muscles[muscle_idx]+'_hdemg_trained_with_DA',
#                                       AE_layers=4, initial_lr=0.00001, distribution_loss_ratio=0.8, lr_update_step=4)
#     DA_model.train_network()
#
#     new_row.append(DA_model.recorded_validation_error_source)
#     new_row.append(DA_model.recorded_validation_error_target)
#     new_row = pd.DataFrame([new_row], columns=['Tested on HDEMG', 'Tested on Delsys without DA',
#                                                'Tested on HDEMG with DA', 'Tested on Delsys with DA'],
#                            index=[list_of_muscles[muscle_idx]])
#
#     muscle_and_subject_specific = pd.concat([muscle_and_subject_specific, new_row], ignore_index=True)
    # muscle_and_subject_specific.to_csv('/media/ag6016/Storage/DomainAdaptation/Reports/muscle_specific_performance.csv')


# INTRA-MUSCLE GENERALISATION ==========================================================================================

# intra_muscle_analysis = pd.DataFrame(columns=['Tested on Quad', 'Tested on Ham', 'Tested on Tibialis', 'Tested on Soleus'])
# for row_idx in range(len(list_of_muscles)):
#     new_row = []  # All the results in this row will be trained on the same muscle group
#     for col_idx in range(len(list_of_muscles)):
#         hdemg_data, hdemg_labels = get_hdemg_data('BS03', list_of_muscles[row_idx], 'Fast1')
#         delsys_data, delsys_labels = get_delsys_data(list_of_muscles[row_idx])
#         hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels,
#                                                                                       delsys_data,
#                                                                                       delsys_labels)
#         training_source_data = list(TCNDataPrep(hdemg_data, hdemg_labels, window_length=512, window_step=40,
#                                                 batch_size=16, sequence_length=15, label_delay=0, training_size=0.8,
#                                                 lstm_sequences=False, split_data=True,
#                                                 shuffle_full_dataset=False).prepped_data)
#
#         training_target_data = list(TCNDataPrep(delsys_data, delsys_labels, window_length=512, window_step=40,
#                                                 batch_size=16, sequence_length=15, label_delay=0, training_size=0.8,
#                                                 lstm_sequences=False, split_data=True,
#                                                 shuffle_full_dataset=False).prepped_data)
#
#         DA_model = RunDomainAdaptationTCN(training_source_data, training_target_data, n_channels=1, epochs=100,
#                                           saved_model_name=list_of_muscles[row_idx] + '_DA_trained',
#                                           AE_layers=4, initial_lr=0.00001, distribution_loss_ratio=0.8,
#                                           lr_update_step=4)
#         DA_model.train_network()
#
#         hdemg_data, hdemg_labels = get_hdemg_data('BS03', list_of_muscles[col_idx], 'Fast1')
#         delsys_data, delsys_labels = get_delsys_data(list_of_muscles[col_idx])
#         hdemg_data, hdemg_labels, delsys_data, delsys_labels = align_signals_together(hdemg_data, hdemg_labels,
#                                                                                       delsys_data,
#                                                                                       delsys_labels)
#         testing_source_data = list(TCNDataPrep(hdemg_data, hdemg_labels, window_length=512, window_step=40,
#                                                batch_size=16, sequence_length=15, label_delay=0, training_size=0.8,
#                                                lstm_sequences=False, split_data=True,
#                                                shuffle_full_dataset=False).prepped_data)
#
#         testing_target_data = list(TCNDataPrep(delsys_data, delsys_labels, window_length=512, window_step=40,
#                                                batch_size=16, sequence_length=15, label_delay=0, training_size=0.8,
#                                                lstm_sequences=False, split_data=True,
#                                                shuffle_full_dataset=False).prepped_data)
#
#         DA_model_fixed = RunDomainAdaptationTCN(testing_source_data, testing_target_data, n_channels=1, epochs=100,
#                                                 saved_model_name=list_of_muscles[row_idx] + '_DA_trained',
#                                                 AE_layers=4, load_model=True, initial_lr=0.00001,
#                                                 distribution_loss_ratio=0.8, lr_update_step=4, fix_weights=True)
#
#         DA_model_fixed.train_network()
#
#         new_row.append(DA_model_fixed.recorded_validation_error_target)
#     new_row = pd.DataFrame([new_row], columns=['Tested on HDEMG', 'Tested on Delsys without DA',
#                                                'Tested on HDEMG with DA', 'Tested on Delsys with DA'],
#                            index=[list_of_muscles[row_idx]])
#
#     intra_muscle_analysis = pd.concat([intra_muscle_analysis, new_row], ignore_index=True)
#     intra_muscle_analysis.to_csv('/media/ag6016/Storage/DomainAdaptation/Reports/intra_muscle_performance.csv')
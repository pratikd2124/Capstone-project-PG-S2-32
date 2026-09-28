import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import torch.multiprocessing
from torch.multiprocessing import Process, Manager, Pool
import ast
import sys
from functools import partial
import json

import os
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import time
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

#import d_hat_neural_networks_July22.exp_script
import d_hat_neural_networks_July22.misc_functions as mf
import d_hat_neural_networks_July22.net_dataset_init_B as ndi
#import d_hat_neural_networks_July22.perform_exp as pe
import d_hat_neural_networks_July22.train_test_loop as ttl
import d_hat_neural_networks_July22.plotting as plotting


#%% Perform each individual EXP, outputting CSV of train vs test losses and plots of errors
      
def perform_EXP(exp_code, n_layers, width, two_sided_outlier_percentage,
                      log_output_data_true, include_XY_dist, include_skwurtoses_true,
                      P_transform, B_transform_true, batch_size,num_epochs, train_frac, 
                      learning_rate, num_exps, XY_symm_true,
                      local_test_dict, loss_string, directories):
    
    rs = np.random.randint(100)
    dhds_train = ndi.D_hat_Dataset(is_train=True, random_seed=rs, 
                               two_sided_outlier_percentage=two_sided_outlier_percentage,
                               log_output_data_true=log_output_data_true,
                               include_XY_dist=include_XY_dist,
                               XY_symm_true = XY_symm_true,
                               local_test_dict = local_test_dict,
                               train_frac=train_frac,
                               include_skwurtoses_true = include_skwurtoses_true,
                               P_transform = P_transform,
                               B_transform_true = B_transform_true,
                               )
    dhds_test = ndi.D_hat_Dataset(is_train=False, random_seed=rs, 
                               two_sided_outlier_percentage=two_sided_outlier_percentage,
                               log_output_data_true=log_output_data_true,
                               include_XY_dist=include_XY_dist,
                               XY_symm_true = XY_symm_true,
                               local_test_dict = local_test_dict,
                               train_frac=train_frac,
                               include_skwurtoses_true = include_skwurtoses_true,
                               P_transform = P_transform,
                               B_transform_true = B_transform_true,
                               )
    n_features = len(dhds_train.df_features.columns)
    
    model = ndi.Net(n_features, n_layers, width)
    
    min_max_col_df = dhds_train.min_max_col_df
    shape_rate_min_max = {'shape_min' : float(min_max_col_df.loc[min_max_col_df['column']=='shape', 'min']),
                          'shape_max' : float(min_max_col_df.loc[min_max_col_df['column']=='shape', 'max']),
                          'rate_min' : float(min_max_col_df.loc[min_max_col_df['column']=='rate', 'min']),
                          'rate_max' : float(min_max_col_df.loc[min_max_col_df['column']=='rate', 'max'])}
    
    # Save metadata of network to JSON 
    
    metadata_dict = {'n_layers': n_layers, 'width':width, 
                     'feature_labels': dhds_train.feature_columns.to_list(), 
                     'output_labels': dhds_train.output_columns.to_list(),
                     'min_max_df': dhds_train.min_max_col_df.to_json(),
                     'P_transform': P_transform}
    with open(directories['network_metadata'] + "/{}.json".format(exp_code), 'w') as outfile:
        json.dump(metadata_dict, outfile)
    
    
    # Train model
    
    train_dataloader = DataLoader(dhds_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dhds_test, batch_size=batch_size, shuffle=True)
    
    if loss_string=="KL_gamma":
        loss_fn = partial(mf.KL, shape_rate_min_max)
    elif loss_string=="MSE":
        loss_fn = nn.MSELoss(reduction='mean')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Have changed to Adam
    
    train_loss_history = [] #each datapoint is an epoch
    test_loss_history = []
    network_epoch_save_path = directories['saved_networks'] + "/{}_nn_epoch_{}".format(exp_code, {})
    network_optimal_save_path = directories['saved_networks'] + "/{}_nn_optimal_epoch".format(exp_code)
    best_test_loss = 9999
    save_increment = local_test_dict['save_increment']
    for t in range(num_epochs):
        print(f"{exp_code} Epoch {t+1}")
        
        save_network = t % save_increment == 0 # save network every e.g. 20 incremements
        
        # If MSE is desired loss function, the sqrt is taken in the train and test loops at the very end
        
        train_mean_loss_per_instance_of_epoch = ttl.train_loop(train_dataloader, model, loss_fn, optimizer, log_output_data_true, loss_string)
        train_loss_history.append(train_mean_loss_per_instance_of_epoch)
        
        #print(train_mean_loss_per_instance_of_epoch)
        
        test_mean_loss_per_instance_of_epoch = ttl.test_loop(test_dataloader, model, loss_fn, 
                                                         t, n_layers, exp_code, log_output_data_true, 
                                                         dhds_test.feature_columns, dhds_test.min_max_col_df, 
                                                         loss_string, directories)
        # I will keep only saving the test data at the optimal epoch so it doesn't generate a million bad plots of the best-middle-worst kind
        test_loss_history.append(test_mean_loss_per_instance_of_epoch)
        
        if save_network:
            torch.save(model.state_dict(), network_epoch_save_path.format(t) + ".pt")
            dummy_input, dummy_output = dhds_train.__getitem__(0)
            torch.onnx.export(model, dummy_input, network_epoch_save_path.format(t) + ".onnx",)
            
        # If test performance is better than previous best, save the network (in both formats)
        if test_mean_loss_per_instance_of_epoch < best_test_loss:
            best_test_loss = test_mean_loss_per_instance_of_epoch
            torch.save(model.state_dict(), network_optimal_save_path + ".pt")
            
            # Save in ONNX format. Get dummy index from training set. 
            # Since idx works on literal (reset) indices, not inherited ones that are weird, the idx is just 0
            dummy_input, dummy_output = dhds_train.__getitem__(0)
            torch.onnx.export(model, dummy_input, network_optimal_save_path + ".onnx", 
                              # verbose=True, 
                              # input_names=dhds_train.feature_columns, 
                              # output_names=dhds_train.output_columns
                              )
        #elif test_mean_loss_per_instance_of_epoch < best_test_loss:
            
            
        
    print("{} done!".format(exp_code))
    
    loss_df = pd.DataFrame({'epoch': np.linspace(1, num_epochs, num_epochs),
                            'train_loss': train_loss_history, 
                            'test_loss': test_loss_history})


    loss_df.to_csv(directories['loss_results'] + "/{}.csv".format(exp_code), index=False)
    
    
    ## Update parameters doc with final train and test data
    params_df = pd.read_csv(directories['exp_name'] + "/parameters_exp_codes.csv", index_col=False)
    
    #print(loss_df)
    
    # Calculate optimal epoch and corresponding loss. Note that this is where test_loss is minimised.
    idx_min_test = loss_df.idxmin()['test_loss']
    
    #print(idx_min_test)
    
    params_df.loc[params_df['exp_code']==exp_code, 'training_loss_opt_epoch'] = loss_df['train_loss'].iloc[idx_min_test]
    params_df.loc[params_df['exp_code']==exp_code, 'test_loss_opt_epoch'] = loss_df['test_loss'].iloc[idx_min_test]
    params_df.loc[params_df['exp_code']==exp_code, 'optimal_epoch'] = loss_df['epoch'].iloc[idx_min_test]
    
    params_df.to_csv(directories['exp_name'] + "/parameters_exp_codes.csv", index=False)
    
    # PLOTTING
    
    # So that each plot has a different colour, also useful for futurue amalgamation plots
    palette_hexes = sns.color_palette('viridis', num_exps).as_hex()
    exp_num = int(exp_code[-3:])
    exp_colour = palette_hexes[exp_num]
    
    if loss_string=="KL_gamma":
        ylim = 1
    elif loss_string=="MSE":
        ylim = 0.1
    
    # Plot train vs test error as function of epoch
    plotting.plot_train_test_vs_epoch(loss_df, exp_colour, loss_string, 
                                      directories['training_plots'], exp_code,
                                      ylim=ylim)
    
    # Plot best middle and worst predictions 
    pt_df = pd.read_csv(directories['pred_vs_truth'] + "/{}.csv".format(exp_code), index_col=False)
    plotting.plot_best_middle_worse_gamma_predictions_driver(pt_df, 
                                                             local_test_dict['is_local'], 
                                                             directories['pred_vs_truth_plots'], 
                                                             exp_code)
   
    # Histogram of loss values 
    plotting.loss_histogram(pt_df, exp_colour, exp_code, directories['loss_histograms'], loss_string)
    
    
    
    
    
    
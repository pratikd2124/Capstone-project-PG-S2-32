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

import os
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import time
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

#import d_hat_neural_networks_July22.exp_script
import d_hat_neural_networks_July22.misc_functions as mf
import d_hat_neural_networks_July22.net_dataset_init_B as ndi
#import d_hat_neural_networks_July22.net_dataset_init as ndi
import d_hat_neural_networks_July22.perform_exp as pe
import d_hat_neural_networks_July22.train_test_loop as ttl
import d_hat_neural_networks_July22.plotting as plotting




#%% Settings of experiment 
directory = os.getcwd()
is_local = directory == "/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/d_hat_neural_networks_July22"

exp_name = "final_test"

exp_name_directory = mf.directory_creator(directory, exp_name)
loss_results_directory = mf.directory_creator(exp_name_directory, "loss_results")
training_plots_directory = mf.directory_creator(exp_name_directory, "training_plots")
pred_vs_truth_directory = mf.directory_creator(exp_name_directory, "pred_vs_truth")
pred_vs_truth_plots_directory = mf.directory_creator(exp_name_directory, "pred_vs_truth_plots")
loss_histograms_directory = mf.directory_creator(exp_name_directory, "loss_histograms")
network_metadata_directory = mf.directory_creator(exp_name_directory, "network_metadata")
saved_networks_directory = mf.directory_creator(exp_name_directory, "saved_networks")
directories = {'exp_name': exp_name_directory,
               'loss_results': loss_results_directory,
               'training_plots': training_plots_directory,
               'pred_vs_truth': pred_vs_truth_directory,
               'pred_vs_truth_plots': pred_vs_truth_plots_directory,
               'loss_histograms': loss_histograms_directory,
               'network_metadata': network_metadata_directory,
               'saved_networks': saved_networks_directory}

#%%
test_run = False  

local_test_dict = {'is_local': is_local, 'test_run': test_run, 
                   'save_increment': 25}

loss_string = 'KL_gamma'

# Set parameters to iterate through
n_layers_list = [15, 20]
width_list = [256, 512]
two_sided_outlier_percentage_list = [1]
include_XY_dist_list=[True]
include_skwurtoses_list = [True]
P_transform_list = ['all_P_asymm'] # usually all_P_asymm
B_transform_true_list = [False]
log_output_data_true_list = [False]
XY_symm_true_list = [False] # Remove the X-Y symmetry cause BAD 

# I think I am unlikely to change these, but included in params nonetheless
batch_size_list = [64]
num_epochs_list = [1000]
train_frac_list = [0.8]
learning_rate_list = [1e-4]

parameter_combinations_df = pd.DataFrame(list(product(XY_symm_true_list,
                                                      log_output_data_true_list, 
                                                      include_XY_dist_list,
                                                      include_skwurtoses_list,
                                                      P_transform_list,
                                                      B_transform_true_list,
                                                      n_layers_list, 
                                                      width_list,
                                                      two_sided_outlier_percentage_list,
                                                      batch_size_list,
                                                      num_epochs_list,
                                                      train_frac_list,
                                                      learning_rate_list,
                                                       ))
                                      ,
                                      columns = ['XY_symm_true',
                                                 'log_output_data_true', 
                                                 'include_XY_dist_true',
                                                 'include_skwurtoses_true',
                                                 'P_transform',
                                                 'B_transform_true',
                                                 'n_layers','width',
                                                 'two_sided_outlier_percentage', 
                                                 'batch_size',
                                                 'num_epochs', 'train_frac',
                                                 'learning_rate',])
exp_codes_list = ['EXP{:03d}'.format(i) for i in range(parameter_combinations_df.shape[0])]
parameter_combinations_df.insert(loc=0, column = 'exp_code', value = exp_codes_list)
parameter_combinations_df['training_loss_opt_epoch'] = 0
parameter_combinations_df['test_loss_opt_epoch'] = 0
parameter_combinations_df['optimal_epoch'] = 0

parameter_combinations_df.to_csv(exp_name_directory + "/parameters_exp_codes.csv", index=False)



#%% TEST RUN
if test_run:
    # #perform_EXP(exp_code, n_layers, width, two_sided_outlier_percentage,
    #                       log_output_data_true, include_XY_dist, include_skwurtoses_true,
    #                       P_transform, B_transform_true, batch_size,num_epochs, train_frac, 
    #                       learning_rate, num_exps, XY_symm_true,
    #                       local_test_dict, loss_string, directories):
    
    pe.perform_EXP('EXP000', 15, 256, 1, False, True, True, 'all_sigma_hat_inverse_B_transform', True, 
                   64, 5, 0.8, 1e-4, 23, False, 
                   local_test_dict, loss_string, directories)
    
else:
#%% Run the damn thing!

    if __name__=='__main__':

        #torch.multiprocessing.set_start_method('spawn')

        begin = time.perf_counter()
        # Setup multiprocessing
        manager = Manager()
        samples = manager.dict()
        if is_local:
            n_cores = 4
        else:
            n_cores = torch.multiprocessing.cpu_count()
        pool = Pool(processes=n_cores)
        jobs = []
        for row_index, pm in parameter_combinations_df.iterrows():

            # Multiprocessing exports CSVs and plots to relevant folders
            # NOTE - the args are in specific spots, so if anything is changed then check here
            begin = time.perf_counter()
            pool.apply_async(pe.perform_EXP, args=(pm['exp_code'],
                                                pm['n_layers'],
                                                pm['width'],
                                                pm['two_sided_outlier_percentage'],
                                                pm['log_output_data_true'],
                                                pm['include_XY_dist_true'],
                                                pm['include_skwurtoses_true'],
                                                pm['P_transform'],
                                                pm['B_transform_true'],
                                                pm['batch_size'],
                                                pm['num_epochs'],
                                                pm['train_frac'],
                                                pm['learning_rate'],
                                                len(parameter_combinations_df.index),
                                                pm['XY_symm_true'], 
                                                local_test_dict,
                                                loss_string,
                                                directories))
            sys.stdout.flush()


        pool.close()
        pool.join() 
        end = time.perf_counter()
        print(f"Total time {end - begin:0.4f} seconds")
        
        # Ensure hyperparameter result df is updated appropriately 
        mf.update_hyperparameter_results(directories)


    #%%
        # Print tables of mean losses for different parameters
        params_df = pd.read_csv(exp_name_directory + "/parameters_exp_codes.csv", index_col=False)
        
        nunique_df = params_df.drop(['exp_code','training_loss_opt_epoch', 'test_loss_opt_epoch', 'optimal_epoch'], axis=1).nunique().to_frame(name='nunique')
        params_changed_list = nunique_df.index[nunique_df['nunique']>1].tolist()

        pd.set_option('display.expand_frame_repr', False)
        for pm in params_changed_list:
            print(params_df.groupby([pm]). mean()[['training_loss_opt_epoch', 'test_loss_opt_epoch', 'optimal_epoch']])





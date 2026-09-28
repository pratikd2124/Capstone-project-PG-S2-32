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
import d_hat_neural_networks_July22.perform_exp as pe
#import d_hat_neural_networks_July22.train_test_loop as ttl
import d_hat_neural_networks_July22.plotting as plotting

#%% Define training and testing loops 

def train_loop(dataloader, model, loss_fn, optimizer, log_output_data_true, loss_string):
    num_instances = len(dataloader.dataset)
    batch_loss_list = []
    batch_size_list = []
    
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if logging data, unlog it now for purposes of loss calculation
        if log_output_data_true:
            pred = torch.exp(pred)-1
            y = torch.exp(y)-1

        # Record loss of batch
        if loss_string=="KL_gamma":
            mean_loss_of_batch = loss.detach().numpy()
        elif loss_string=="MSE":
            mean_loss_of_batch = loss.item()
        batch_loss_list.append(mean_loss_of_batch)
        batch_size_list.append(len(pred))
    
    # Note that num_instances == sum(batch_size_list)
    
    total_loss_of_epoch = np.dot(batch_loss_list, batch_size_list)
    mean_loss_per_instance_of_epoch = total_loss_of_epoch / num_instances
    if loss_string=="KL_gamma":
        return mean_loss_per_instance_of_epoch
    elif loss_string=="MSE":
        return np.sqrt(mean_loss_per_instance_of_epoch)
    
    


def test_loop(dataloader, model, loss_fn, epoch_num, n_layers, exp_code, log_output_data_true,
              feature_columns, min_max_col_df, loss_string, directories):
    num_instances = len(dataloader.dataset)
    batch_loss_list = []
    batch_size_list = []
    

    with torch.no_grad():
        prediction_list = []
        true_y_list = []
        X_list = []
        for X, y in dataloader:
            
            # Record truth and prediction 
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # if logging data, unlog it now for purposes of loss calculation
            if log_output_data_true:
                pred = torch.exp(pred)-1
                y = torch.exp(y)-1
                
            prediction_list.append(pred)
            true_y_list.append(y)

            X_list.append(X)
            
            # Record loss of batch
            if loss_string=="KL_gamma":
                mean_loss_of_batch = loss.detach().numpy()
            elif loss_string=="MSE":
                mean_loss_of_batch = loss.item()
            batch_loss_list.append(mean_loss_of_batch)
            batch_size_list.append(len(pred))

            
    total_loss_of_epoch = np.dot(batch_loss_list, batch_size_list)
    mean_loss_per_instance_of_epoch = total_loss_of_epoch / num_instances

    # Create dataframe of predictions vs truth
    prediction_list_cat_tensor = torch.cat(prediction_list).numpy()
    true_y_list_cat_tensor = torch.cat(true_y_list).numpy()
    X_list_tensor = torch.cat(X_list).numpy()
    X_df = pd.DataFrame(X_list_tensor, columns = feature_columns)
    

    prediction_vs_truth_df_epoch = pd.DataFrame({'shape_pred': prediction_list_cat_tensor[:,0],
                                           'shape_truth': true_y_list_cat_tensor[:,0],
                                           'rate_pred': prediction_list_cat_tensor[:,1],
                                           'rate_truth': true_y_list_cat_tensor[:,1],
                                           'epoch_num':epoch_num
                                           })
    prediction_vs_truth_df_epoch = X_df.join(prediction_vs_truth_df_epoch)
    
    pred_vals = torch.tensor(prediction_vs_truth_df_epoch[['shape_pred', 'rate_pred']].values)
    truth_vals = torch.tensor(prediction_vs_truth_df_epoch[['shape_truth', 'rate_truth']].values)
    
    if loss_string == "KL_gamma":
        prediction_vs_truth_df_epoch['loss'] = loss_fn(pred_vals, truth_vals, reduce_mean=False) 
    elif loss_string == "MSE":
        # This is the old sqrt average MSE of instance of in epoch 
        prediction_vs_truth_df_epoch['loss'] = np.sqrt((prediction_vs_truth_df_epoch['shape_pred'] - prediction_vs_truth_df_epoch['shape_truth'])**2 + (prediction_vs_truth_df_epoch['rate_pred'] - prediction_vs_truth_df_epoch['rate_truth'])**2)
    
    # Renormalise values of features and outputs
    for col in min_max_col_df['column']:
        
        min_val = float(min_max_col_df.loc[min_max_col_df['column']==col, 'min'])
        max_val = float(min_max_col_df.loc[min_max_col_df['column']==col, 'max'])
        
        if col in ['shape', 'rate']:
            prediction_vs_truth_df_epoch[col + '_pred'] = (max_val-min_val) * prediction_vs_truth_df_epoch[col + '_pred'] + min_val
            prediction_vs_truth_df_epoch[col + '_truth'] = (max_val-min_val) * prediction_vs_truth_df_epoch[col + '_truth'] + min_val
        else:
            prediction_vs_truth_df_epoch[col] = (max_val-min_val) * prediction_vs_truth_df_epoch[col] + min_val
    
    
    pred_truth_path = directories['pred_vs_truth'] + "/{}.csv".format(exp_code)
    
    if epoch_num==0:
        prediction_vs_truth_df_epoch.to_csv(pred_truth_path, index=False)
    else:
        # Only add to pred_vs_truth_df if current epoch is better than last.
        # This is because we only care about looking at the neural network performance 
        # on the optimal epoch anyway, and it was blowing the sizes out by too much. 
        prediction_vs_truth_df = pd.read_csv(pred_truth_path, index_col=False)
        mean_loss_old = prediction_vs_truth_df['loss'].mean()
        mean_loss_new = prediction_vs_truth_df_epoch['loss'].mean()
        
        if mean_loss_new < mean_loss_old:
            prediction_vs_truth_df_epoch.to_csv(pred_truth_path, index=False)
            # Otherwise leave as old epoch 
            
    if loss_string=="KL_gamma":
        return mean_loss_per_instance_of_epoch
    elif loss_string=="MSE":
        return np.sqrt(mean_loss_per_instance_of_epoch)










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
import scipy

import os
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import time
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

plt.rcParams["patch.force_edgecolor"] = True

#import d_hat_neural_networks_July22.exp_script
import d_hat_neural_networks_July22.misc_functions as mf
import d_hat_neural_networks_July22.net_dataset_init as ndi
import d_hat_neural_networks_July22.perform_exp as pe
import d_hat_neural_networks_July22.train_test_loop as ttl
#import d_hat_neural_networks_July22.plotting as plotting

#%% Train and test loss vs epoch
def plot_train_test_vs_epoch(loss_df, exp_colour, loss_string, 
                             plots_directory, exp_code, ylim=None, change_text=None):
   
    ## Plot train and test error
    fig, ax = plt.subplots(1,1)
    ax.cla()
    
    sns.lineplot(data=loss_df, x='epoch', y='train_loss', color=exp_colour, label='Train', linestyle='--')
    sns.lineplot(data=loss_df, x='epoch', y='test_loss', color=exp_colour, label='Test')
    
    if loss_string=="KL_gamma":
        plt.title('{} - Mean KL loss vs epoch'.format(exp_code))
    elif loss_string=="MSE":
        plt.title('{} - Sqrt of MSE loss vs epoch'.format(exp_code))
    
    if ylim:
        plt.ylim([0, ylim])

    plt.xlabel('Epoch')
    plt.ylabel('Mean instance loss of epoch')
    fig.tight_layout()
    plt.close()
    
    if change_text:
        fig.savefig(plots_directory + "/{0}_{1}.png".format(change_text, exp_code))
    else:
        fig.savefig(plots_directory + "/{}.png".format(exp_code))
    

#%% Plot train test with different ylim to what was chosen in exp run

num_exps = 4
palette_hexes = sns.color_palette('viridis', num_exps).as_hex()
loss_df_directory = "/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/d_hat_neural_networks_July22/final_V1/loss_results"
plots_directory = loss_df_directory[:-13] + "/training_plots"

for filename in os.listdir(loss_df_directory):
    f = os.path.join(loss_df_directory, filename)
    loss_df =pd.read_csv(f, index_col=False)    
    
    exp_code = filename[3:-4]
    exp_num = int(exp_code)
    exp_colour = palette_hexes[exp_num]
    
    plot_train_test_vs_epoch(loss_df, exp_colour, loss_string='KL_gamma', 
                                  plots_directory = plots_directory, exp_code="EXP" + exp_code, 
                                  ylim=0.2, change_text='zoom_in')


    
    
#%% Plot best, middle, worse predictions as gamma curves

def plot_best_middle_worse_gamma_predictions(df, predicted_colour, exp_name, 
                                             pt_exp_plot_directory, bmw_string):
    
    n_x = 10000
    x_pdf = np.linspace(0.001, 1.0, n_x)
    
    for i, row in df.iterrows():
        
        shape_pred = row['shape_pred']
        scale_pred = 1.0/row['rate_pred'] # scipy uses scale = 1/rate
        shape_truth = row['shape_truth']
        scale_truth = 1.0/row['rate_truth']
        
        
            
        y_pred_pdf = scipy.stats.gamma.pdf(x_pdf, a = shape_pred, scale = scale_pred)
        y_truth_pdf = scipy.stats.gamma.pdf(x_pdf, a = shape_truth, scale = scale_truth)
        
        pdf = pd.DataFrame({'x': x_pdf, 'y': y_pred_pdf, 'pt': 'Predicted'})
        tdf = pd.DataFrame({'x': x_pdf, 'y': y_truth_pdf, 'pt': 'Truth'})
        data_df = pd.concat([pdf, tdf], ignore_index=True)
        
        fig, ax = plt.subplots(1)
        p = sns.lineplot(data=data_df, x='x', y='y', hue='pt', ax=ax, 
                         hue_order = ['Truth', 'Predicted'], palette=['#FF0000', predicted_colour ])
        p.legend_.set_title(None)
        p.set_title('Gamma PDF of Predicted vs Truth \n{0}: loss = {1:.6g}'.format(bmw_string, row['loss']))
        
        
        plt.close()
        fig.savefig(pt_exp_plot_directory + "/{0}_loss={1:.6g}.png".format(bmw_string, row['loss']))
    
    


def plot_best_middle_worse_gamma_predictions_driver(pt_df, is_local, pt_plot_directory, exp_name):

    # Recall that pt_df now only has the optimal epoch in it
    
    pt_df.reset_index(inplace=True)
    middle_index = int(pt_df.shape[0]/2)
    
    best_df = pt_df.sort_values('loss', ascending=True).iloc[:10]
    middle_df = pt_df.sort_values('loss', ascending=True).iloc[middle_index-5:middle_index+5]
    worst_df = pt_df.sort_values('loss', ascending=True).iloc[-10:]
    
    # Create folder for plots
    pt_exp_plot_directory = mf.directory_creator(pt_plot_directory, exp_name)
    
    plot_best_middle_worse_gamma_predictions(best_df, '#FDE725FF', exp_name, pt_exp_plot_directory, 'Best')
    plot_best_middle_worse_gamma_predictions(middle_df, '#22A884FF', exp_name, pt_exp_plot_directory, 'Middle')
    plot_best_middle_worse_gamma_predictions(worst_df, '#440154FF', exp_name, pt_exp_plot_directory, 'Worst')



#%% Plot dist of loss values 

def loss_histogram(pt_df, colour, exp_name, plot_directory, loss_string="KL_gamma"):
    
    fig, ax = plt.subplots(1)
    
    loss_mean = pt_df['loss'].mean()
    loss_median = pt_df['loss'].median()
    
    p = sns.histplot(data=pt_df, x="loss", binwidth=0.002, color=colour)
    if loss_string=="KL_gamma":
        p.set_xlim([0, 0.1])
    elif loss_string=="MSE":
        p.set_xlim([0, 0.15])
    p.set_title("{} - Histogram of loss at optimal epoch".format(exp_name))
    plt.axvline(x=loss_mean, color="#DF488D", label="Mean")
    plt.axvline(x=loss_median, color="#EFCC98", label="Median")
    plt.legend()
    
    plt.close()
    fig.savefig(plot_directory + "/{}.png".format(exp_name))


#%% Plot loss histograms with different ylim to what was chosen in exp run

# num_exps = 16
# palette_hexes = sns.color_palette('viridis', num_exps).as_hex()
# loss_df_directory = "/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/d_hat_neural_networks_July22/refactored_6/pred_vs_truth"
# plots_directory = loss_df_directory[:-13] + "/loss_histograms"

# for filename in os.listdir(loss_df_directory):
#     f = os.path.join(loss_df_directory, filename)
#     pt_df =pd.read_csv(f, index_col=False)    
    
#     exp_code = filename[3:-4]
#     exp_num = int(exp_code)
#     exp_colour = palette_hexes[exp_num]
    
#     loss_histogram(pt_df, exp_colour, exp_name="EXP" + exp_code, plot_directory = plots_directory, loss_string='KL_gamma')



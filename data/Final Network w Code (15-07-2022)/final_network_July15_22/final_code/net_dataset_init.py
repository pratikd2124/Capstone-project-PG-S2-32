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
#import d_hat_neural_networks_July22.net_dataset_init as ndi
import d_hat_neural_networks_July22.perform_exp as pe
import d_hat_neural_networks_July22.train_test_loop as ttl
import d_hat_neural_networks_July22.plotting as plotting

#%% Neural Network definition

class Net(nn.Module):

    def __init__(self, n_features, n_layers, width):
        super(Net, self).__init__()
        
        # Set up layers of nn. NOTE - widths are a fixed value.
        inner_layers_list = [[('linear{}'.format(l), nn.Linear(width, width)), 
                             ('relu{}'.format(l), nn.ReLU())] for l in range(2,n_layers+1)]
        inner_layers_list = [i for i in chain.from_iterable(inner_layers_list)]
        od = OrderedDict([('linear1', nn.Linear(n_features, width)),
                          ('relu1', nn.ReLU())] + inner_layers_list +
                         [('linear_out', nn.Linear(width, 2))])
                          
        # Define nn
        self.linear_relu_stack = nn.Sequential(od)
        
    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y


#%% Define d_hat_dataset class 
class D_hat_Dataset(Dataset):
    def __init__(self, is_train, random_seed, two_sided_outlier_percentage, log_output_data_true, 
                 include_XY_dist, XY_symm_true, local_test_dict, train_frac=0.8, 
                 include_skwurtoses_true = True, P_transform='all_P_asymm', B_transform_true = False):
        # two_sided_outlier_percentage corresponds to the percentage removed in each tail - e.g. 5 would be bounds of [5, 95]
        

        if local_test_dict['is_local']:
            dfs_folder_direct = "/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/d_hat_neural_networks_July22/instance_data"
        else:
            dfs_folder_direct = "/home/ubuntu/gemfinder-storage/d_hat_P_matrix_csvs" # NEED TO FIX ON LAMBDA LABS
        
        
        dfs_list = []
        for filename in os.listdir(dfs_folder_direct):
            f = os.path.join(dfs_folder_direct, filename)
            dfs_list.append(pd.read_csv(f, index_col=False))
        df_combined = pd.concat(dfs_list, ignore_index=True)

        # Reduce dataset size for speed when testing
        if local_test_dict['test_run']:
            df_combined = df_combined.sample(n=1000)
        
        # Some kurtosis and skewness values are NaN in dataframe, so remove these instances 
        df_combined.dropna(inplace=True)

        # Translate "c(1,2,3,4)" P_matrix from R into features for neural network 
        df_combined = mf.P_matrix_transform(df_combined, transform_type = P_transform)
        
        # Remove the XY symmetrised rows if the data contains them
        if not XY_symm_true:
            unique_indices = np.sort(np.unique(df_combined['exp_code'], return_index=True)[1])
            df_combined = df_combined.iloc[unique_indices,].reset_index(drop=True)
            
        # Remove unwanted features (I JUST MOVED THIS HOPEFULLY IT WORKS)
        feature_drop_list = []
        if not include_XY_dist: 
            feature_drop_list = feature_drop_list + ['X_{}'.format(i) for i in range(1,5)] + ['Y_{}'.format(i) for i in range(1,5)]
        if not include_skwurtoses_true:
            feature_drop_list = feature_drop_list + ['mean_{}'.format(i) for i in ['X', 'Y']] + ['stdev_{}'.format(i) for i in ['X', 'Y']] + ['skewness_{}'.format(i) for i in ['X', 'Y']] + ['kurtosis_{}'.format(i) for i in ['X', 'Y']]
        
        df_combined = df_combined.drop(feature_drop_list, axis=1)
        
        # Normalise data, min-max method
        non_normalise_list = ['exp_code']
        
        # NOW NORMALISING ALL VARIABLES FOR EASE
        min_max_col_df = pd.DataFrame({'column' : df_combined.drop(non_normalise_list,axis=1).columns})
        min_max_col_df['min'] = min_max_col_df.apply(lambda col: min(df_combined[col['column']]), axis=1)
        min_max_col_df['max'] = min_max_col_df.apply(lambda col: max(df_combined[col['column']]), axis=1)
        for col in min_max_col_df['column']:
            min_val = float(min_max_col_df.loc[min_max_col_df['column']==col, 'min'])
            max_val = float(min_max_col_df.loc[min_max_col_df['column']==col, 'max'])
            df_combined[col] = (df_combined[col] - min_val)/(max_val-min_val)
        
        
        # Remove outside of two_sided_outlier_percentage quantile (both shape and rate)
        sz = df_combined['shape'].size-1
        df_combined['shape_quantile'] = df_combined['shape'].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
        df_combined['rate_quantile'] = df_combined['rate'].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
        
        lower_quantile = two_sided_outlier_percentage
        upper_quantile = 100 - two_sided_outlier_percentage
        df_combined = df_combined[(df_combined.shape_quantile>lower_quantile) & (df_combined.shape_quantile<upper_quantile)]
        df_combined = df_combined[(df_combined.rate_quantile>lower_quantile) & (df_combined.rate_quantile<upper_quantile)]
        
        
        # Log data - applies transformation log(x+1)
        if log_output_data_true:
            df_combined['shape'] = np.log(df_combined['shape']+1)
            df_combined['rate'] = np.log(df_combined['rate']+1)
           
        
        # Train vs test split
        np.random.seed(random_seed)
        num_total_samples = len(df_combined)
        num_train = round(train_frac*num_total_samples)
        sample_list = np.array((range(num_total_samples)))
        train_indices = np.random.choice(sample_list, 
                                         size=num_train,
                                         replace=False)
        test_indices = np.array(list(set(sample_list) - set(train_indices)))
        if is_train:
            df_combined = df_combined.iloc[train_indices,]
        else:
            df_combined = df_combined.iloc[test_indices,]
            
        # This line is dangerous and could be a bug further down the line
        np.random.seed()
        
        col_drop_list = ['exp_code','shape', 'rate', 'shape_quantile', 'rate_quantile']
        self.df_features = df_combined.drop(col_drop_list, axis=1)
        
        self.feature_columns = self.df_features.columns
        
        self.min_max_col_df = min_max_col_df
    
        self.df_output = df_combined[['shape', 'rate']]
        self.output_columns = self.df_output.columns
        
        # Extract dummy index - keep this consistent and do on training set
        #self.dummy_idx = self.df_features.index[0]


    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, idx):
        features_idx = torch.tensor(self.df_features.iloc[idx,:].values.astype(np.float32))
        output_idx = torch.tensor(self.df_output.iloc[idx,:].values.astype(np.float32))

        return features_idx, output_idx
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

# import d_hat_neural_networks_July22.exp_script
# #import d_hat_neural_networks_July22.misc_functions as mf
# import d_hat_neural_networks_July22.net_dataset_init as ndi
# import d_hat_neural_networks_July22.perform_exp as pe
# import d_hat_neural_networks_July22.train_test_loop as ttl
# import d_hat_neural_networks_July22.plotting as plotting
#%%
def directory_creator(directory, new_subdir):
    # Creates a new directory if it doesn't already exist
    
    new_directory = directory + "/" + new_subdir
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        
    return new_directory

def update_hyperparameter_results(directories):
    params_df = pd.read_csv(directories['exp_name'] + "/parameters_exp_codes.csv", index_col=False)
    for i, row in params_df.iterrows():
        exp_code = row['exp_code']
        loss_direct = directories['loss_results'] + "/{}.csv".format(exp_code)
        if os.path.isfile(loss_direct):
            loss_df = pd.read_csv(loss_direct, index_col=False)
        
            idx_min_test = loss_df.idxmin()['test_loss']
            params_df.loc[params_df['exp_code']==exp_code, 'training_loss_opt_epoch'] = loss_df['train_loss'].iloc[idx_min_test]
            params_df.loc[params_df['exp_code']==exp_code, 'test_loss_opt_epoch'] = loss_df['test_loss'].iloc[idx_min_test]
            params_df.loc[params_df['exp_code']==exp_code, 'optimal_epoch'] = loss_df['epoch'].iloc[idx_min_test]
            
    params_df.to_csv(directories['exp_name'] + "/parameters_exp_codes_update.csv", index=False)
    

#%% Functions for B_matrix transform

def moment_calculator(prob_dist, dist_name, num_types=4):
    # Takes normalised prob distribution prob_dist on 1:4, i.e. X or Y
    
    ### THIS IS NOW USELESS ###
    
    rv_vec = np.arange(1, num_types+1) # i.e. 1,2,3,4
    
    if prob_dist.ndim==1:
        # in the case prob_dist = np.array([1,2,3,4]) etc.
        mean = np.dot(rv_vec, prob_dist)
        stdev = np.sqrt(np.dot((rv_vec - mean)**2, prob_dist))
        skewness = np.dot((rv_vec - mean)**3, prob_dist)/stdev**3
        kurtosis = np.dot((rv_vec - mean)**4, prob_dist)/stdev**4
      
        moments = pd.Series({'mean_{}'.format(dist_name): mean,
                             'stdev_{}'.format(dist_name): stdev,
                             'skewness_{}'.format(dist_name): skewness,
                             'kurtosis_{}'.format(dist_name): kurtosis})
        
    # FALSE! Cannot calculate moments of the transformed prob_dist (3-dim) because it isn't an actual prob_dist! 
    # else:
    #     # in the case prob_dist.shape = (num_instances, num_types) e.g. (1000, 3)
    #     mean = np.dot(rv_vec, prob_dist.T) # produces (1000, )
    #     mean_long = mean[:, np.newaxis]
        
    #     rv_vec_long = np.tile(rv_vec, (prob_dist.shape[0], 1))
    #     stdev = np.sqrt(np.dot((rv_vec_long - mean_long)**2, prob_dist))
        
        
    return moments

def sigma_maker(j, P_matrix):
    # creates the sigma_j covariance matrix in the d_hat formula
    rho_j = P_matrix[j, :]
    sigma_j = -1*np.outer(rho_j, rho_j) # produces 4x4 matrix of (4,1) x (1,4)
    np.fill_diagonal(sigma_j, rho_j*(1-rho_j))
    
    return sigma_j
    
def B_matrix_transform(df, is_local, P_transform):
    # Takes raw output of R generated instances (so all features should be present, e.g. X_1, X_2, X_3, X_4)

    if is_local:
        B_matrix = np.loadtxt("/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/d_hat_neural_networks_July22/B_matrix.txt", 
                    comments="#", delimiter=" ", unpack=False)
    else:
        None
    
    # Transform raw X, Y dists to 3-dim using B @ X.T (which is (3,4) x (4, num_instances))
    X_vecs = df[['X_{}'.format(i) for i in range(1, 5)]].to_numpy()
    X_B_vecs = np.transpose(B_matrix @ np.transpose(X_vecs)) # produces shape of (num_instances x 3) 
    Y_vecs = df[['Y_{}'.format(i) for i in range(1, 5)]].to_numpy()
    Y_B_vecs = np.transpose(B_matrix @ np.transpose(Y_vecs))
    
    # Calculate absolute difference
    XmY_B_array = np.abs(X_B_vecs - Y_B_vecs)

    # Calculate new moments, only going up to skewness (since kurtosis can't exist for a dist on 1:3)
    # FALSE! Cannot calculate moments of the transformed prob_dist (3-dim) because it isn't an actual prob_dist! 
    #mf.moment_calculator(prob_dist, dist_name, num_types=4)
    
    #drop moments implicitly below
    
    # Euclid_dist
    euclid_dist_XY_B = np.sqrt(np.sum((X_B_vecs - Y_B_vecs)**2, axis=1))
    
    # deal with P matrix

    df = P_matrix_transform(df, transform_type = P_transform, B_matrix=B_matrix)
    
    # Paste it all together
    
    shBi_cols = ['shBi_{}'.format(i) for i in range(9)] # since shBi is a (3,3) matrix
    new_columns_list = ['exp_code'] + ['X_B_{}'.format(i) for i in range(1,4)] + ['Y_B_{}'.format(i) for i in range(1,4)] + ['XmY_B_{}'.format(i) for i in range(1,4)] + ['euclid_dist_XY_B', 'n_X', 'n_Y'] + shBi_cols + ['shape', 'rate']
    df_new = pd.DataFrame(np.concatenate([df['exp_code'][:, np.newaxis],
                                          X_B_vecs,
                                          Y_B_vecs,
                                          XmY_B_array,
                                          euclid_dist_XY_B[:, np.newaxis],
                                          df[['n_X', 'n_Y'] + shBi_cols + ['shape', 'rate']],
                                          ], axis=1),
                          columns = new_columns_list)
    
    return df_new

#%% Functions for P_matrix conversion

def asymm_calc(P):
    # MSE of non-diagonal entries of a 4x4 matrix 
    
    non_diag_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)] # since py has 0 index
    asymm_total = 0
    for pair in non_diag_pairs:
        asymm_total += (P[pair[0], pair[1]] - P[pair[1], pair[0]])**2
    return asymm_total / len(non_diag_pairs)


def R_matrix_to_np(df_row, transform_type, B_matrix):
    # Takes R_matrix from df row and converts it into features
    R_matrix = df_row['P_matrix']
    
    py_list = "[" + R_matrix[2:-1] + "]"
    py_list = np.array(ast.literal_eval(py_list))
    P_matrix = np.reshape(py_list,(4,4)).T
    
    if transform_type == 'diag_asymm':
        P_matrix_features = pd.Series({'diag_1': P_matrix[0,0], 
                                       'diag_2': P_matrix[1,1], 
                                       'diag_3': P_matrix[2,2], 
                                       'diag_4': P_matrix[3,3], 
                                       'asymm': asymm_calc(P_matrix)
            })
    elif transform_type == 'all_P_asymm':
        P_dict = {'P_{}'.format(i): P_matrix.flat[i] for i in range(len(P_matrix.flat))}
        P_dict['asymm'] = asymm_calc(P_matrix)
        P_matrix_features = pd.Series(P_dict)
    elif transform_type == 'all_sigma_hat_inverse_B_transform':
        # Record all sigma_hat values (and not asymm because it doesn't make sense) under B transform 
        # i.e. (B @ sigma_hat @ B^T)^-1
        
        sigma_list = [sigma_maker(j, P_matrix) for j in range(4)]
        # Calculate sigma_hat = \sum_{i=1}^4 (X_i/n + Y_i/m) \sigma_i
        # which supposes X and Y are normalised (so features from the df)
        sigma_hat_list = [(df_row['X_{}'.format(j)]/df_row['n_X'] + df_row['Y_{}'.format(j)]/df_row['n_Y']) * sigma_list[j-1] for j in range(1,5)]
        sigma_hat = sum(sigma_hat_list) # interestingly np.sum sums everything and produces a scalar, this produces a matrix
        
        # print(sigma_hat)
        # print(B_matrix)
        
        # shBi = sigma hat B inverse = (B @ sigma_hat @ B^T)^-1
        shBi = np.linalg.inv(B_matrix @ sigma_hat @ B_matrix.T)
        
        # flatten output, don't bother with asymm as it is now meaningless
        P_matrix_features = pd.Series({'shBi_{}'.format(i): shBi.flat[i] for i in range(len(shBi.flat))})
        
        
    return P_matrix_features


def P_matrix_transform(df, transform_type, B_matrix=False):
    # Takes df with P_matrix in "c(1,2,3,4)" form and converts to df with P_matrix features
    
    # Extract features from each row of df
    df_new = df.apply(lambda row: R_matrix_to_np(row, transform_type, B_matrix), axis=1, result_type='expand')
    
    # Delete P_matrix col of df
    df = df.drop(['P_matrix', 'P_matrix_question'], axis=1)
    
    # Insert new features into df
    i = df.columns.get_loc("shape") # column index of shape
    df = pd.concat([df.iloc[:, :i],
                    df_new,
                    df.iloc[:,i:]], axis=1)
    
    return df

#%% New KL between Gaussian loss function

def K(shape_1, rate_1, shape_2, rate_2):
    # Calculate K quantity according to that stack overflow post (calculation is vectorised)
    
    # This was to check if the code was working when there were negative values. 
    # Work it did! It would start with negative predictions and then the bias would lead it away from that. 
    # if sum(shape_1<0)>0 or sum(rate_1<0)>0:
    #     print('negative predictions')
    
    # Ensure negatives are turned positive to make logs fine (this is then overridden later)
    shape_1 = torch.abs(shape_1)
    rate_1 = torch.abs(rate_1)
    
    # Set values according to stack post
    a = 1/rate_1
    b = shape_1
    c = 1/rate_2
    d = shape_2
    
    K_calc = -c*d/a - b*torch.log(a) - torch.lgamma(b) + (b-1)*(torch.digamma(d)+torch.log(c))

    return K_calc

    

def KL(srmm, output, target, reduce_mean = True):
    # Takes model_output and target of batch (normalised), calculates KL between Gamma distributions
    # srmm = shape_rate_min_max dict for appropriate renormalisation
    
    # Renormalise shape and rate for KL calculation
    shape_p = (srmm['shape_max'] - srmm['shape_min'])*output[:,0] + srmm['shape_min']
    rate_p = (srmm['rate_max'] - srmm['rate_min'])*output[:,1] + srmm['rate_min']
    shape_t = (srmm['shape_max'] - srmm['shape_min']) * target[:,0] + srmm['shape_min']
    rate_t = (srmm['rate_max'] - srmm['rate_min']) * target[:,1] + srmm['rate_min']
    
    #print(srmm)
    
    K_tt = K(shape_t, rate_t, shape_t, rate_t) 
    K_pt = K(shape_p, rate_p, shape_t, rate_t)
    K_diff = K_tt - K_pt
    
    #print(K_diff)
    
    # Ensures any negative values of K_diff are biased against with loss proportional to square difference (so differentiation makes some sense)
    neg_shape_pred_boolean = shape_p<0
    neg_rate_pred_boolean = rate_p<0
    
    nss = sum(neg_shape_pred_boolean)
    nrs = sum(neg_rate_pred_boolean)
    # if nss>0 or nrs>0:
    #     print(nss)
    #     print(nrs)
    
    K_diff[neg_shape_pred_boolean] = 5*(shape_p[neg_shape_pred_boolean] - shape_t[neg_shape_pred_boolean])**2
    K_diff[neg_rate_pred_boolean] = 5*(rate_p[neg_rate_pred_boolean] - rate_t[neg_rate_pred_boolean])**2

    #print(K_diff)
    # Reduce to scalar or keep vector of individual losses
    if reduce_mean:
        KL_loss = torch.mean(K_diff)
        
        #if torch.isnan(KL_loss):
            #print(K_pt)
            #print(K_diff)
    else:
        # Returns vector of individual values
        KL_loss = K_diff
        
    
    
    return KL_loss
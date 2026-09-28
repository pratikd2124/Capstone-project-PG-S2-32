import numpy as np
import json 
import pandas as pd

#%%
# Read in P_matrices and network metadata
with open("/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/RY_P_matrices/P_matrices_test.json") as f:
    P_matrices = json.load(f)
    
with open("/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/d_hat_neural_networks_July22/test_save_networks/network_metadata/EXP003.json") as f:
    metadata = json.load(f)

#%% Necessary functions for calculation

def moment_calculator(prob_dist, dist_name, num_types=4):
    # Takes normalised prob distribution prob_dist on 1:4, i.e. X or Y
    rv_vec = np.arange(1, num_types+1) # i.e. 1,2,3,4
    
    mean = np.dot(rv_vec, prob_dist)
    stdev = np.sqrt(np.dot((rv_vec - mean)**2, prob_dist))
    skewness = np.dot((rv_vec - mean)**3, prob_dist)/stdev**3
    kurtosis = np.dot((rv_vec - mean)**4, prob_dist)/stdev**4
  
    moments = pd.Series({'mean_{}'.format(dist_name): mean,
                         'stdev_{}'.format(dist_name): stdev,
                         'skewness_{}'.format(dist_name): skewness,
                         'kurtosis_{}'.format(dist_name): kurtosis})
    return moments

def asymm_calc(P):
    # MSE of non-diagonal entries of a 4x4 matrix 
    
    non_diag_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)] # since py has 0 index
    asymm_total = 0
    for pair in non_diag_pairs:
        asymm_total += (P[pair[0], pair[1]] - P[pair[1], pair[0]])**2
    return asymm_total / len(non_diag_pairs)


def transform_P_matrix(question, P_transform):
    # Takes R_matrix from df row and converts it into features
    
    P_matrix = np.array(P_matrices[question])
    
    if P_transform == 'diag_asymm':
        P_matrix_features = pd.Series({'diag_1': P_matrix[0,0], 
                                       'diag_2': P_matrix[1,1], 
                                       'diag_3': P_matrix[2,2], 
                                       'diag_4': P_matrix[3,3], 
                                       'asymm': asymm_calc(P_matrix)
            })
    if P_transform == 'all_P_asymm':
        P_dict = {'P_{}'.format(i): P_matrix.flat[i] for i in range(len(P_matrix.flat))}
        P_dict['asymm'] = asymm_calc(P_matrix)
        P_matrix_features = pd.Series(P_dict)
    
    
    return P_matrix_features

#%% Calculate features

# Given data from portal query: 
    # x - vector of 4 entries with integer values (number of students selecting each response) 
    # y - ditto
    # question - string of RY question code 


def extract_features(x, y, question, P_transform = 'all_P_asymm'):
    # Normalise classes 
    n_X = x.sum()
    n_Y = y.sum()
    X = x/n_X
    Y = y/n_Y
    
    # Calculate absolute difference
    XmY = abs(X-Y)
    
    # Calculate euclid distance between X and Y 
    euclid_dist_XY = np.round(np.sqrt(np.sum((X - Y)**2)),4)
    
    # Features series (in order)
    X_series = pd.Series({'X_{}'.format(i+1) : X[i] for i in range(4)})
    Y_series = pd.Series({'Y_{}'.format(i+1) : Y[i] for i in range(4)})
    XmY_series = pd.Series({'XmY_{}'.format(i+1) : XmY[i] for i in range(4)})
    X_moments_series = moment_calculator(X, 'X', num_types=4)
    Y_moments_series = moment_calculator(Y, 'Y', num_types=4)
    euclid_dist_XY_series = pd.Series({'euclid_dist_XY': euclid_dist_XY})
    n_series = pd.Series({'n_X': n_X, 'n_Y': n_Y})
    P_matrix_features_series = transform_P_matrix(question, P_transform)
    
    final_features = pd.concat([X_series,
                                Y_series,
                                XmY_series,
                                X_moments_series,
                                Y_moments_series,
                                euclid_dist_XY_series,
                                n_series,
                                P_matrix_features_series])
    
    return final_features

#%% 
# e.g. 
x = np.array([40, 52, 7, 14])
y = np.array([9, 17, 54, 12])
question = 'ry9'
P_transform = metadata['P_transform']

final_features = extract_features(x,y,question,P_transform)

#%% Normalisation and renormalisation

mm_df = pd.read_json(metadata['min_max_df']) # min_max_col_df

def normalise(feature_value, fmin, fmax):
    return float((feature_value - fmin)/(fmax - fmin))

def reverse_normalise(feature_value, fmin, fmax):
    return float((fmax - fmin) * feature_value + fmin)

# This apply stuff is only specific to pandas - you just need some way of applying the noramlisation 
# procedure across all values using the min_max_col_df values
ff_df = final_features.to_frame('value') # Feature names are the index

# Normalise features using mm_df
for row_index, row in ff_df.iterrows():
    ff_df.loc[row_index, 'value'] = normalise(row['value'], 
                                              mm_df.loc[mm_df['column']==row_index, 'min'],
                                              mm_df.loc[mm_df['column']==row_index, 'max'])
    
# Ensure features are columns (so df is one row), thus can transform into tensor or vector or whatever
ff_df = ff_df.transpose() 

# Finally, after receiving output from model (shape, rate) (in that order), renormalise. 
# Example output: 
out_df = pd.DataFrame([0.08, 0.192], index=['shape', 'rate'], columns = ['value'])

# Reverse normalisation using mm_df
for row_index, row in out_df.iterrows():
    out_df.loc[row_index, 'value'] = reverse_normalise(row['value'], 
                                              mm_df.loc[mm_df['column']==row_index, 'min'],
                                              mm_df.loc[mm_df['column']==row_index, 'max'])


    






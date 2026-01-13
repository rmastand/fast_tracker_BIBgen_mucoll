import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

epsilon = 1e-12
def logit_transform(x, all_min, all_max, cushion ):
    
    x_norm = (x-all_min)/(all_max-all_min)
    x_norm = (1.0 - 2.0*cushion)*x_norm + cushion
    logit_arguments = (x_norm/(1.0-x_norm+epsilon)) + epsilon
    
    num_invalid_entries = sum(logit_arguments <= 0)
    if num_invalid_entries > 0:
        print("Invalid log. Try again with larger cushion")
        return None
    else:
        logit = np.log(x_norm/(1.0-x_norm+epsilon) + epsilon)
        return logit



def preprocess_data(X):

    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
        
    X_preproc = minmax_scaler.fit_transform(X)
    for i in range(X.shape[1]):
        X_preproc[:,i] = logit_transform(X_preproc[:,i], 0.0, 1.0, 0.0)
    X_preproc = standard_scaler.fit_transform(X_preproc)

    with open("minmax", "wb") as ofile:
        pickle.dump(minmax_scaler, ofile)
    with open("standard", "wb") as ofile:
        pickle.dump(standard_scaler, ofile)
        
    return X_preproc

def inverse_preprocess_data(X_preproc):

    with open("minmax", "rb") as ifile:
        minmax_scaler = pickle.load(ifile)
    with open("standard", "rb") as ifile:
        standard_scaler = pickle.load(ifile)

    X = standard_scaler.inverse_transform(X_preproc)
    X = np.exp(X) / (1.0 + np.exp(X))    
    X = minmax_scaler.inverse_transform(X)

    return X

def unscale_mass(scaled_x, SB_left, SB_right):
    
    unscaled_x =  scaled_x*preproc_info["std"] + preproc_info["mean"]
    #inverse logit
    x_norm = np.exp(unscaled_x) / (1.0 + np.exp(unscaled_x))
    x_norm = (x_norm - 0.01) / 0.98
    
    return x_norm*(preproc_info["max"]-preproc_info["min"]) + preproc_info["min"]


def clean_data(x):
    
    remove_nan =  x[~np.isnan(x).any(axis=1)]
    remove_inf = remove_nan[~np.isinf(remove_nan).any(axis=1)]
    
    return remove_inf

def bootstrap_array(data_array, seed):
    np.random.seed(seed)
    indices_to_take = np.random.choice(range(data_array.shape[0]), size = data_array.shape[0], replace = True) 
    return data_array[indices_to_take]


def assemble_banded_datasets(data_dict, feature_set, bands):
    
    banded_data = {}
    
    for b in bands:
        num_events_band = data_dict[b]["dimu_mass"].shape[0]
        events_band = np.empty((num_events_band, len(feature_set)))
        for i, feat in enumerate(feature_set):
            # default test set
            events_band[:,i] = data_dict[b][feat].reshape(-1,)
        banded_data[b] = events_band
        
    return banded_data

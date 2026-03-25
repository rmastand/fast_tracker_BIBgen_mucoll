import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

epsilon = 1e-12


def load_in_data(collection_list, features, working_dir, training_frac):
        
    data = []
    context = []
    
    for i, collection in enumerate(collection_list):
        tmp_data = np.load(f"{working_dir}/npys/nuGun_pT_0_50/{collection}_SimTrackerHit.npy")
        tmp_context = int(i)*np.ones((int(len(tmp_data)*training_frac),1))
        context.append(tmp_context)
        data.append(tmp_data[:int(len(tmp_data)*training_frac)])

        
    
    data = np.vstack(data)
    data[:,0] = np.log(data[:,0])
    context = np.vstack(context)

    if features == "rphi":
        r = np.sqrt(data[:, 1]**2 + data[:, 2]**2)
        theta = np.arctan2(data[:, 2], data[:, 1])
        data[:,1] = r
        data[:,2] = theta
        feature_labels = ["log($E$) [Gev]", "$r$", "$\phi$", "$z$", "$t$ [s]", "context"]
    else: 
        feature_labels = ["log($E$) [Gev]", "$x$", "$y$", "$z$", "$t$ [s]", "context"]


    return data, context, feature_labels



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



def preprocess_data(X, flow_training_dir, ZUKO_ID):
    """
    Preprocess data without modifying the original array.
    Applies log to the first column and standardizes all columns.
    """

    if ZUKO_ID in ["UNAF", "NCSF"]:
        min_max_scaler = MinMaxScaler(feature_range=(-3,3))
        X_preproc = min_max_scaler.fit_transform(X)
        with open(f"{flow_training_dir}/minmax", "wb") as ofile:
            pickle.dump(min_max_scaler, ofile)


    else:
        standard_scaler = StandardScaler()
        X_preproc = standard_scaler.fit_transform(X)
        with open(f"{flow_training_dir}/standard", "wb") as ofile:
            pickle.dump(standard_scaler, ofile)
    

    return X_preproc


def inverse_preprocess_data(X_preproc, flow_training_dir, ZUKO_ID):
    """
    Inverse preprocessing without modifying the input array.
    Inverts standardization and applies exp to the first column.
    """
    # load scaler
    if ZUKO_ID in ["UNAF", "NCSF"]:
        with open(f"{flow_training_dir}/minmax", "rb") as ifile:
            min_max_scaler = pickle.load(ifile)
        X = min_max_scaler.inverse_transform(X_preproc)

    else:
        with open(f"{flow_training_dir}/standard", "rb") as ifile:
            standard_scaler = pickle.load(ifile)
        X = standard_scaler.inverse_transform(X_preproc)
        
    

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


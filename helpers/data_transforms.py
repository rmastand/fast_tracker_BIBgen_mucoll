import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

epsilon = 1e-12


num_sectors = {
    "InnerTrackerEndcapCollection": 26, 
    "OuterTrackerEndcapCollection": 48, 
    "VertexEndcapCollection": 16,
}

def load_in_data(collection_list, features, working_dir, training_frac, num_cond_features=0, feature_order=None, num_files=1):
        
    data = []
    context = []
    
    for i, collection in enumerate(collection_list):
        for r in range(num_files):

            if num_cond_features == 0:
                tmp_data = np.load(f"{working_dir}/npys/nuGun_pT_0_50/{collection}_SimTrackerHit_reco_{r}.npy")
                data.append(tmp_data[:int(len(tmp_data)*training_frac)])
            elif num_cond_features > 0:
                tmp_data = np.load(f"{working_dir}/npys/nuGun_pT_0_50/{collection}_SimTrackerHit_conditional_reco_{r}.npy")

                data.append(tmp_data[:int(len(tmp_data)*training_frac), :-num_cond_features])
                context.append(tmp_data[:int(len(tmp_data)*training_frac), -num_cond_features:])
               
        
    
    data = np.vstack(data)
    if num_cond_features > 0:
        context = np.vstack(context)
    data[:,0] = np.log(data[:,0]) #preprocess the energy
    print

    if features == "rphi":
        r = np.sqrt(data[:, 1]**2 + data[:, 2]**2)
        phi = np.arctan2(data[:, 2], data[:, 1])
        data[:,1] = r
        data[:,2] = phi

        # delta_phi = 2*np.pi / num_sectors[collection]
        # sector_index = np.floor(phi / delta_phi)
        # sector_coord = sector_index * delta_phi
        # phi_local = phi - sector_coord
        # data[:,2] = phi_local
        # sinphi = np.sin(phi)
        # cosphi = np.cos(phi)
        # # replace phi with sin and cos
        # data = np.hstack([data[:,:2], sinphi[:, np.newaxis], cosphi[:, np.newaxis], data[:,3:]])
        feature_labels = ["log($E$) [Gev]", "$r$", "$\phi$", "$z$", "$t$", "side", "layer", "module", "sensor"]
    else:
        feature_labels = ["log($E$) [Gev]", "$x$", "$y$", "$z$", "$t$ [s]", "side", "layer", "module", "sensor"]

    X = np.hstack([data,  context]) if num_cond_features > 0 else data

    if feature_order is not None:
        X = X[:, feature_order]
        feature_labels = [feature_labels[i] for i in feature_order]

    for i in range(num_cond_features):
        feature_labels[-1-i] = feature_labels[-1-i] + " (cond)"

    return X, feature_labels



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



def preprocess_data(X, flow_training_dir, ZUKO_ID, num_cond_features=0):
    """
    Preprocess data without modifying the original array.
    Applies log to the first column and standardizes all columns.
    Does not preprocess the context features if they are present.
    """

    if num_cond_features > 0:
        X_to_preproc = X[:,:-num_cond_features]
        X_context = X[:,-num_cond_features:]
    else:
        X_to_preproc = X

    if ZUKO_ID in ["UNAF", "NCSF"]:
        min_max_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_preproc = min_max_scaler.fit_transform(X_to_preproc)
        with open(f"{flow_training_dir}/minmax", "wb") as ofile:
            pickle.dump(min_max_scaler, ofile)


    else:
        standard_scaler = StandardScaler()
        X_preproc = standard_scaler.fit_transform(X_to_preproc)
        with open(f"{flow_training_dir}/standard", "wb") as ofile:
            pickle.dump(standard_scaler, ofile)


    
    

    return np.hstack([X_preproc, X_context]) if num_cond_features > 0 else X_preproc


def inverse_preprocess_data(X_preproc, flow_training_dir, ZUKO_ID, num_cond_features=0):
    """
    Inverse preprocessing without modifying the input array.
    Inverts standardization and applies exp to the first column.
    """
   
    if num_cond_features > 0:
        X_to_unpreproc = X_preproc[:,:-num_cond_features]
        X_context = X_preproc[:,-num_cond_features:]
    else:
        X_to_unpreproc = X_preproc

        


    if ZUKO_ID in ["UNAF", "NCSF"]:
        with open(f"{flow_training_dir}/minmax", "rb") as ifile:
            min_max_scaler = pickle.load(ifile)
        X = min_max_scaler.inverse_transform(X_to_unpreproc)

    else:
        with open(f"{flow_training_dir}/standard", "rb") as ifile:
            standard_scaler = pickle.load(ifile)
        X = standard_scaler.inverse_transform(X_to_unpreproc)
        
    

    return np.hstack([X, X_context]) if num_cond_features > 0 else X

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


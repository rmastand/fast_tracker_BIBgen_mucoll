import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

epsilon = 1e-12


num_sectors = {
    "InnerTrackerEndcapCollection": 26, 
    "OuterTrackerEndcapCollection": 48, 
    "VertexEndcapCollection": 16,
    "OuterTrackerBarrelCollection": 164,
}


def load_in_data(collection_list, features, working_dir, training_frac, num_cond_features=0, feature_order=None, num_files=1, use_local_phi=False):
        
    data = []
    context = []
    layers = []
    
    for i, collection in enumerate(collection_list):
        for r in range(num_files):

            
            
            if num_cond_features == 0:
                tmp_data = np.load(f"{working_dir}/npys/nuGun_pT_0_50/{collection}_SimTrackerHit_reco_{r}.npy")
                
                data.append(tmp_data[:int(len(tmp_data)*training_frac)])
            elif num_cond_features > 0:
                tmp_data = np.load(f"{working_dir}/npys/nuGun_pT_0_50/{collection}_SimTrackerHit_conditional_reco_{r}.npy")

                data.append(tmp_data[:int(len(tmp_data)*training_frac), :-num_cond_features])
                context.append(tmp_data[:int(len(tmp_data)*training_frac), -num_cond_features:])
                layers.append(tmp_data[:int(len(tmp_data)*training_frac),7])
               
        
    
    data = np.vstack(data)
    layers = np.vstack(layers).reshape(-1)
    if num_cond_features > 0:
        context = np.vstack(context)
    data[:,0] = np.log(data[:,0]) #preprocess the energy

    if features == "rphi":
        r = np.sqrt(data[:, 1]**2 + data[:, 2]**2)
        phi = np.arctan2(data[:, 2], data[:, 1])
        data[:,1] = r
        data[:,2] = phi

        if use_local_phi:

            num_modules_per_layer = {0: 92, 1: 128, 2: 164}
            for l in range(3):
                layer_mask = (layers == l)
                phi_l = phi[layer_mask]
                n = num_modules_per_layer[l]
                delta_phi = 2 * np.pi / n
                sector_index = np.floor(phi_l / delta_phi).astype(int)
                sector_coord = sector_index * delta_phi
                phi_local = phi_l - sector_coord  # in [0, delta_phi)
                data[layer_mask, 2] = phi_local / delta_phi * 0.1  # normalize to [0, 10)

            # import matplotlib.pyplot as plt
            # print(data[:, 2])
            # print(np.unique(layers))
            # plt.figure()
            # plt.hist(data[:, 2], bins = 100)
            # plt.savefig("test")

            # x = r*np.cos(data[:, 2])
            # y = r*np.sin(data[:, 2])

            # plt.figure(figsize=(10,10))
            # plt.scatter(x, y, s = 0.01)
            # plt.xlim(700,1600)
            # plt.ylim(-50,200)
            # plt.savefig("test2")
                
            feature_labels = ["log($E$) [Gev]", "$r$", "$\phi$ (local)", "$z$", "$t$", "system", "side", "layer", "module", "sensor"]

        else:
            feature_labels = ["log($E$) [Gev]", "$r$", "$\phi$", "$z$", "$t$", "system", "side", "layer", "module", "sensor"]
    else:
        feature_labels = ["log($E$) [Gev]", "$x$", "$y$", "$z$", "$t$ [s]", "system", "side", "layer", "module", "sensor"]

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




def inverse_preprocess_data_torch(X_preproc, flow_training_dir, ZUKO_ID, num_cond_features=0):
    import torch

    if num_cond_features > 0:
        X_to_unpreproc = X_preproc[:, :-num_cond_features]
        X_context = X_preproc[:, -num_cond_features:]
    else:
        X_to_unpreproc = X_preproc
        X_context = None

    device = X_preproc.device
    dtype = X_preproc.dtype

    if ZUKO_ID in ["UNAF", "NCSF"]:
        with open(f"{flow_training_dir}/minmax", "rb") as ifile:
            scaler = pickle.load(ifile)

        data_min = torch.tensor(scaler.data_min_, device=device, dtype=dtype)
        data_max = torch.tensor(scaler.data_max_, device=device, dtype=dtype)

        feature_min = -np.pi
        feature_max = np.pi

        X = (X_to_unpreproc - feature_min) / (feature_max - feature_min)
        X = X * (data_max - data_min) + data_min

    else:
        with open(f"{flow_training_dir}/standard", "rb") as ifile:
            scaler = pickle.load(ifile)

        mean = torch.tensor(scaler.mean_, device=device, dtype=dtype)
        scale = torch.tensor(scaler.scale_, device=device, dtype=dtype)

        X = X_to_unpreproc * scale + mean

    if num_cond_features > 0:
        return torch.cat([X, X_context], dim=1)

    return X
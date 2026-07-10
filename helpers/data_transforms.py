import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

epsilon = 1e-12

import numpy as np

from helpers.material_map import (
    OUTER_LAYERS,
    INNER_LAYERS,
    VERTEX_LAYERS,
    VERTEX_SUPPORT_THICKNESS,
)


def wrap_to_pi(phi):
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def get_nphi(collection, layer):
    layer = int(layer)

    if collection == "OuterTrackerBarrelCollection":
        return OUTER_LAYERS[layer]["nphi"]

    if collection == "InnerTrackerBarrelCollection":
        return 2 * INNER_LAYERS[layer]["nphi_half"]

    if collection == "VertexBarrelCollection":
        return VERTEX_LAYERS[layer]["nstaves"]

    if collection == "OuterTrackerEndcapCollection":
        return 48

    if collection == "InnerTrackerEndcapCollection":
        return 26

    if collection == "VertexEndcapCollection":
        return 16

    raise ValueError(f"Unknown collection: {collection}")


def local_phi_transformation(
    X,
    layers,
    phi_index,
    collection,
    direction="forward",
    r_col=1,
    phi_col=2,
    local_scale=0.1,
):
    """
    Convert global phi <-> local phi using the stored module/sensor phi index.

    Assumes X is already in r-phi coordinates:
        X[:, r_col]   = r
        X[:, phi_col] = phi

    For barrel collections:
        phi_index should be the module index.

    For endcap collections:
        phi_index should be the sensor/sector index.

    For VertexBarrelCollection:
        local phi is actually a local tangential stave coordinate,
        because vertex barrel staves are flat ladders, not angular wedges.

    direction:
        "forward": global phi -> local phi
        "reverse": local phi -> global phi
    """

    X = X.copy()

    layers = np.asarray(layers).reshape(-1)
    phi_index = np.asarray(phi_index).reshape(-1).astype(int)

    if len(layers) != len(X):
        raise ValueError("layers must have the same length as X")

    if len(phi_index) != len(X):
        raise ValueError("phi_index must have the same length as X")

    for layer in np.unique(layers):
        layer = int(layer)
        mask = layers == layer

        if not np.any(mask):
            continue

        nphi = get_nphi(collection, layer)
        delta_phi = 2.0 * np.pi / nphi

        module = phi_index[mask] % nphi
        phi_center = module * delta_phi

        # ------------------------------------------------------------
        # Special case: vertex barrel flat staves
        # ------------------------------------------------------------
        if collection == "VertexBarrelCollection":
            layer_info = VERTEX_LAYERS[layer]

            width = layer_info["width"]
            half_width = width / 2.0
            offset = layer_info["offset"]

            r = X[mask, r_col]

            if direction == "forward":
                phi = X[mask, phi_col]

                # Tangential coordinate relative to stave center.
                # For a point (r, phi), projection on local tangent is:
                #     tangent = r * sin(phi - phi_center) - offset
                tangent = r * np.sin(wrap_to_pi(phi - phi_center)) - offset

                # Normalize to roughly [-local_scale, local_scale]
                X[mask, phi_col] = tangent / half_width * local_scale

            elif direction == "reverse":
                local = X[mask, phi_col]
                tangent = local / local_scale * half_width

                # Preserve r and solve:
                #     tangent = r * sin(phi - phi_center) - offset
                arg = (tangent + offset) / np.maximum(r, 1e-12)
                arg = np.clip(arg, -1.0, 1.0)

                phi = phi_center + np.arcsin(arg)
                X[mask, phi_col] = wrap_to_pi(phi)

            else:
                raise ValueError("direction must be 'forward' or 'reverse'")

            continue

        # ------------------------------------------------------------
        # All other collections: angular local coordinate
        # ------------------------------------------------------------
        if direction == "forward":
            phi = X[mask, phi_col]
            dphi = wrap_to_pi(phi - phi_center)

            # Normalize one sector width to roughly local_scale
            X[mask, phi_col] = dphi / delta_phi * local_scale

        elif direction == "reverse":
            local = X[mask, phi_col]
            dphi = local / local_scale * delta_phi

            phi = phi_center + dphi
            X[mask, phi_col] = wrap_to_pi(phi)

        else:
            raise ValueError("direction must be 'forward' or 'reverse'")

    return X

    
def load_in_data(collection_list, features, working_dir, training_frac, num_cond_features=0, feature_order=None, num_files=1, use_local_phi=False):
        
    X = []
    layers = []
    phi_index = []
    
    for i, collection in enumerate(collection_list):
        for r in range(num_files):

            
            
            if num_cond_features == 0:
                tmp_data = np.load(f"{working_dir}/npys/nuGun_pT_0_50/{collection}_SimTrackerHit_reco_{r}.npy")
                
                X.append(tmp_data[:int(len(tmp_data)*training_frac)])
            elif num_cond_features > 0:
                tmp_data = np.load(f"{working_dir}/npys/nuGun_pT_0_50/{collection}_SimTrackerHit_conditional_reco_{r}.npy")

                X.append(tmp_data[:int(len(tmp_data)*training_frac)])

                # side, layer, module, sensor starting from index 6
                layers.append(tmp_data[:int(len(tmp_data)*training_frac),7])
                if "Barrel" in collection:
                    phi_index.append(tmp_data[:int(len(tmp_data)*training_frac),8]) # module defines the phi
                elif "Endcap" in collection:
                    phi_index.append(tmp_data[:int(len(tmp_data)*training_frac),9]) # sensor defines the phi
               
        
    
    X = np.vstack(X)
    layers = np.concatenate(layers).reshape(-1)
    phi_index = np.concatenate(phi_index).reshape(-1)

    
   
    X[:,0] = np.log(X[:,0]) #preprocess the energy

    if features == "rphi":
        r = np.sqrt(X[:, 1]**2 + X[:, 2]**2)
        phi = np.arctan2(X[:, 2], X[:, 1])
        X[:,1] = r
        X[:,2] = phi

       
        

        if use_local_phi:

            # import matplotlib.pyplot as plt
        
            # plt.figure()
            # plt.hist(X[:, 2], bins = 100)
            # plt.show()
    
            # x = r*np.cos(X[:, 2])
            # y = r*np.sin(X[:, 2])
    
            # plt.figure(figsize=(15,15))
            # plt.scatter(x, y, s = 0.001)
            # # plt.xlim(700,1600)
            # # plt.ylim(-50,200)
            # plt.show()

            X = local_phi_transformation(
                X,
                layers=X[:, 7],
                phi_index=X[:, 8] if "Barrel" in collection else X[:, 9],
                collection=collection,
                direction="forward",
            )
            # import matplotlib.pyplot as plt
        
            # plt.figure()
            # plt.hist(X[:, 2], bins = 100)
            # plt.show()
    
            # x = r*np.cos(X[:, 2])
            # y = r*np.sin(X[:, 2])
    
            # plt.figure(figsize=(10,10))
            # plt.scatter(x, y, s = 0.01)
            # plt.gca().set_aspect("equal", adjustable="box")

            # # plt.xlim(700,1600)
            # # plt.ylim(-50,200)
            # plt.show()

            # X = local_phi_transformation(
            #     X,
            #     layers=X[:, 7],
            #     phi_index=X[:, 8] if "Barrel" in collection else X[:, 9],
            #     collection=collection,
            #     direction="reverse",
            # )
            # import matplotlib.pyplot as plt
        
            # plt.figure()
            # plt.hist(X[:, 2], bins = 100)
            # plt.show()
    
            # x = r*np.cos(X[:, 2])
            # y = r*np.sin(X[:, 2])
    
            # plt.figure(figsize=(15,15))
            # plt.scatter(x, y, s = 0.001)
            # plt.gca().set_aspect("equal", adjustable="box")

            # # plt.xlim(700,1600)
            # # plt.ylim(-50,200)
            # plt.show()

                
            feature_labels = ["log($E$) [Gev]", "$r$ [mm]", "$\phi$ (local)", "$z$ [mm]", "$t$ [s]", "system", "side", "layer", "module", "sensor"]

        else:
            feature_labels = ["log($E$) [Gev]", "$r$ [mm]", "$\phi$", "$z$ [mm]", "$t$ [s]", "system", "side", "layer", "module", "sensor"]
    else:
        feature_labels = ["log($E$) [Gev]", "$x$ [mm]", "$y$ [mm]", "$z$ [mm]", "$t$ [s]", "system", "side", "layer", "module", "sensor"]


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



def preprocess_data(X, flow_training_dir, ZUKO_ID, num_cond_features=0, scaler_exists=False):
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
        if scaler_exists:
            with open(f"{flow_training_dir}/minmax", "rb") as ifile:
                min_max_scaler = pickle.load(ifile)
            X_preproc = min_max_scaler.transform(X_to_preproc)
            
        else:
            min_max_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
            X_preproc = min_max_scaler.fit_transform(X_to_preproc)
            with open(f"{flow_training_dir}/minmax", "wb") as ofile:
                pickle.dump(min_max_scaler, ofile)
    

    else:
        if scaler_exists:
            with open(f"{flow_training_dir}/standard", "rb") as ifile:
                standard_scaler = pickle.load(ifile)
            X_preproc = standard_scaler.inverse_transform(X_to_preproc)
            
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
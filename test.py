import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import torch
import argparse
import yaml
from numba import cuda

from helpers.density_estimator import DensityEstimator
from helpers.ANODE_training_utils import train_ANODE, plot_ANODE_losses
from helpers.data_transforms import preprocess_data 

from helpers.flow_sampling import get_flow_samples


SEED = 8
BATCH_SIZE = 1024
NUM_FEATURES = 5
EPOCHS = 1000
PATIENCE = 50


# computing
device = cuda.get_current_device()
device.reset()
torch.set_num_threads(2)
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)
seed = int(SEED)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str)
args = parser.parse_args()


path_to_config_file = f"configs/{args.config}.yml"
flow_training_dir = f"/pscratch/sd/r/rmastand/muon/models/{args.config}"
os.makedirs(flow_training_dir, exist_ok=True)



collections_TrackerHitPlane = [
    "IBTrackerHits",
    "IBTrackerHitsConed",
    "IETrackerHits", 
    "IETrackerHitsConed",
    "OBTrackerHits",
    "OBTrackerHitsConed", 
    "OETrackerHits",        
    "OETrackerHitsConed",          
    "VBTrackerHits",               
    "VBTrackerHitsConed",   
    "VETrackerHits",  
    "VETrackerHitsConed",        
]

collections_SimTrackerHit = [
   # "InnerTrackerBarrelCollection",
   # "InnerTrackerBarrelCollectionConed",
   # "InnerTrackerEndcapCollection",
    #"InnerTrackerEndcapCollectionConed", 
    "OuterTrackerBarrelCollection",     
   # "OuterTrackerBarrelCollectionConed",
  #  "OuterTrackerEndcapCollection",  
   # "OuterTrackerEndcapCollectionConed",
   # "VertexBarrelCollection",   
   # "VertexBarrelCollectionConed",   
   # "VertexEndcapCollection",
   # "VertexEndcapCollectionConed",
]

def logit_scale(x):
    return np.log(x / (1.0 - vx + epsilon) + epsilon)



data = []

for collection in collections_SimTrackerHit: # TODO coned too?
    
    tmp = np.load(f"/pscratch/sd/r/rmastand/muon/npys/{collection}_SimTrackerHit.npy")
    data.append(tmp)

data = np.vstack(data)[:,[0,2,3,4,5]]
num_cond_inputs = 0

# preprocessing from CATHODE paper
X = preprocess_data(data)
# add a random noise feature for now
#X = np.hstack([X,  np.random.normal(size=(len(X),1))])

if True:
    # train val split
    from sklearn.model_selection import train_test_split
    
    data_train, data_val = train_test_split(X, test_size=0.2, random_state=42)
    
    print(f"Train data has shape {data_train.shape}.")
    print(f"Val data has shape {data_val.shape}.")

    
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers = 8, pin_memory = True)
        
    
    """
    CREATE THE FLOW
    """
    
    anode = DensityEstimator(path_to_config_file, NUM_FEATURES, device=device,
                             verbose=False, bound=False)
    model, optimizer = anode.model, anode.optimizer
    
 
    train_ANODE(model, optimizer, train_loader, val_loader, f"flow",
                EPOCHS, PATIENCE, savedir=flow_training_dir, device=device, num_cond_inputs=num_cond_inputs, verbose=True, no_logit=False, data_std=None)






# plot losses
train_losses = np.load(os.path.join(flow_training_dir, f"flow_train_losses.npy"))
val_losses = np.load(os.path.join(flow_training_dir, f"flow_val_losses.npy"))
val_losses = val_losses[train_losses < 1e20]
train_losses = train_losses[train_losses < 1e20]
plot_ANODE_losses(train_losses, val_losses, yrange=None,
savefig=os.path.join(flow_training_dir, f"loss_plot"),suppress_show=True)


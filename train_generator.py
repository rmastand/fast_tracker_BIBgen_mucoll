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
#from helpers.flow_sampling import get_flow_samples

from helpers.models.diffusion.diffusion_model import ScoreNet1D, marginal_prob_std, loss_fn
import functools


SEED = 8
BATCH_SIZE = 1024
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
parser.add_argument("-id", "--id", type=str)
parser.add_argument("-flow", "--train_flow_model", action="store_true")
parser.add_argument("-diff", "--train_diffusion_model", action="store_true")
parser.add_argument("-context", "--num_cond_inputs", type=int, default=0)

args = parser.parse_args()


path_to_config_file = f"configs/{args.config}.yml"
flow_training_dir = f"/pscratch/sd/r/rmastand/muon/models/{args.config}/{args.id}"
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
    "InnerTrackerBarrelCollection",
   # "InnerTrackerBarrelCollectionConed",
    "InnerTrackerEndcapCollection",
    #"InnerTrackerEndcapCollectionConed", 
    "OuterTrackerBarrelCollection",     
   # "OuterTrackerBarrelCollectionConed",
    "OuterTrackerEndcapCollection",  
   # "OuterTrackerEndcapCollectionConed",
    "VertexBarrelCollection",   
   # "VertexBarrelCollectionConed",   
    "VertexEndcapCollection",
   # "VertexEndcapCollectionConed",
]

def logit_scale(x):
    return np.log(x / (1.0 - vx + epsilon) + epsilon)



data = []
context = []

for i, collection in enumerate(collections_SimTrackerHit): # TODO coned too?

    tmp_data = np.load(f"/pscratch/sd/r/rmastand/muon/npys/{collection}_SimTrackerHit.npy")
    tmp_context = int(i)*np.ones((tmp_data.shape[0],1))
    context.append(tmp_context)
    data.append(tmp_data)



data = np.vstack(data)[:,[0,2,3,4,5]]
context = np.vstack(context)


# preprocessing from CATHODE paper
X = preprocess_data(data, flow_training_dir)

if args.num_cond_inputs == 1:
    # add a random noise feature for now
    X = np.hstack([X,  context])
elif args.num_cond_inputs == 0:
    pass
else:
    print("ERROR")
    exit()

NUM_COND_INPUTS = args.num_cond_inputs
NUM_FEATURES =  X.shape[1] - NUM_COND_INPUTS

print(f"num features: {NUM_FEATURES}. num context: {NUM_COND_INPUTS}")

print(X.shape)

# train val split
from sklearn.model_selection import train_test_split

data_train, data_val = train_test_split(X, test_size=0.2, random_state=42)

print(f"Train data has shape {data_train.shape}.")
print(f"Val data has shape {data_val.shape}.")


train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8, pin_memory = True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers = 8, pin_memory = True)
    

if args.train_flow_model:
    anode = DensityEstimator(path_to_config_file, NUM_FEATURES, device=device,
                             verbose=False, bound=False)
    model, optimizer = anode.model, anode.optimizer
    
    
    train_ANODE(model, optimizer, train_loader, val_loader, f"flow",
                EPOCHS, PATIENCE, savedir=flow_training_dir, device=device, num_cond_inputs=NUM_COND_INPUTS, verbose=True, no_logit=False, data_std=None)
    
elif args.train_diffusion_model: # TODO PATIENCE

    with open(path_to_config_file, "r") as ifile:
        configs_dict = yaml.safe_load(ifile)


    from torch.optim import Adam
    import tqdm.notebook as notebook
    

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=configs_dict["sigma"], device=device)

    score_model = torch.nn.DataParallel(ScoreNet1D(marginal_prob_std=marginal_prob_std_fn, channels=configs_dict["channels"], embed_dim=configs_dict["embed_dim"]))
    score_model = score_model.to(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Assuming you have your model defined as 'model'
    num_params = count_parameters(score_model)
    print(f"Number of trainable parameters: {num_params}")

    optimizer = Adam(score_model.parameters(), lr=configs_dict["optimizer"]["lr"], weight_decay=configs_dict["optimizer"]["weight_decay"])
    tqdm_epoch = notebook.trange(EPOCHS)
    

    train_losses, val_losses = [], []
    best_val_loss = 1e10
    for epoch in tqdm_epoch:
        
        avg_train_loss, avg_val_loss = 0.0, 0.0
        num_items = 0

        # train
        for x in train_loader:
            x = x.to(device).float()  
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_train_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_train_loss / num_items))
        train_losses.append(avg_train_loss / num_items)


        # val
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device).float()   
                loss = loss_fn(score_model, x, marginal_prob_std_fn)
                avg_val_loss += loss.item() * x.shape[0]

            val_losses.append(avg_val_loss / num_items)

        if val_losses[-1] < best_val_loss:
            print("new best val loss", best_val_loss )
            best_val_loss = val_losses[-1]
            
            # Update the checkpoint after each epoch of training.
            torch.save(score_model.state_dict(), f"{flow_training_dir}/ckpt.pth")

        np.save(f"{flow_training_dir}/flow_train_losses.npy", train_losses)
        np.save(f"{flow_training_dir}/flow_val_losses.npy", val_losses)


    

      



# plot losses
train_losses = np.load(os.path.join(flow_training_dir, f"flow_train_losses.npy"))
val_losses = np.load(os.path.join(flow_training_dir, f"flow_val_losses.npy"))
val_losses = val_losses[train_losses < 1e20]
train_losses = train_losses[train_losses < 1e20]
plot_ANODE_losses(train_losses, val_losses, yrange=None,
savefig=os.path.join(flow_training_dir, f"loss_plot"),suppress_show=True)


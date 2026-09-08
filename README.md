# GenBIB

This repository contains code for training generative models, generating GenBIB samples, and evaluating them for a muon collider detector.

## Running scripts

### Model training

#### Setup

Before running any of the scripts in this repository, you'll want to make a version of `configs_train.yaml` for your setup. Relevant keywords to set are:

- `PATH_TO_DATA_DIR`: path to wherever your `.slcio` files are on your local machine *these can be downloaded from ])
- `PATH_TO_OUTPUT_DIR`: path where all model training intermediates will be saved to
- `PATH_TO_WANDB_DIR`: path where all wandb training intermediates will be saved to
- `FEATURE_ORDER`: specify an order for feature training observables. Ensure that all conditioning observables are last. The available observables are: `{0: energy, 1: x, 2: y, 3: z, 4: t, 5: system, 6: side, 7: layer, 8: module, 9: sensor}`
- `NUM_COND_INPUTS`: number of conditioning observables
- `FEATURE_INDICES_DICT`: a dictionary that maps between the ordered indices in `FEATURE_ORDER` to the actual features. If you train in the `r-phi` basis, you can generally map $x \rightarrow r$ and $y \rightarrow \phi$
  For example, with `FEATURE_ORDER = [0, 4, 1, 2, 3, 6, 7, 8, 9]` in the `r-phi` basis, the reordered columns are `[log(E), t, r, phi, z, side, layer, module, sensor]`.

#### Scripts

1. `01_process_data.ipynb`: a notebook mainly for data exploration of the `.slcio` files and to convert them to `.npy` arrays for ML model training. Note that you have the option to save out a different set of observables than `["Edep", "x", "y", "z", "t", "system", "side", "layer", "module", "sensor"]` if you so choose.
2. `02_generate_samples.py`: the main training script. Here, you'll have the option to choose whether you train the `flow` or `tabddpm` model with the flag `--MODEL`. We recommend training (`--TRAIN`) and sampling (`--EVAL`) in separate training runs, since the sampling process can take quite a bit of time depending on your machine.
3. `03_evaluate_samples.py`: evaluates the quality of the ML models hits by training BDTs to discriminate the full simulation BIB from the GenBIB

For convenience, we also have a notebook `nice_plots.ipynb`, which was used to make all of the plots in the accompanying paper.

### Track fitting

The GenBIB `.npy` files produced in the previous section are used as inputs to the track-reconstruction pipeline below.

#### Setup

The next set of scripts need to be run on a machine with the muon collider software image available. Details of how to access the image are [here](https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2024/). The scripts in this repository use version 2.11.

First you'll need to place the GenBIB `.npy` files into a folder `RECOBIB_FLOW_SAMPLES_DIR/folder_name`. Within the folder, the files should be named after the corresponding collection, e.g., `OuterTrackerBarrelCollection.npy`. Then `folder_name` will be the first argument of the `run_pipeline` method in `osg/run_osg.sh`.

You'll also want to make version of `osg/configs_osg.yaml` for your setup. Relevant keywords to set are:

- `SCRATCH_DIR`: path where the MuonColliderSoftware is installed
- `SLCIO_DIR`: path where all intermediate `.slcio` files (e.g. digitization and reconstruction intermediates) will be scored
- `RECOBIB_FLOW_SAMPLES_DIR`: defined above
- `GEO_MAP_PATH`: path where youw want to save the detector geometry (x,y,z) $\rightarrow$ cellID function (this will be made automatically in `numpy_to_lcio.py`)
- `COMPARE_OUTPUT_DIR`: path to store any plots make in `compare_tracks.py`


#### Scripts

All of the scripts are in the `osg` folder in this repository. We have compiled them into a helpful bash script `osg/run_osg.sh`, which runs 3 sets of commands:

1. `numpy_to_lcio.py`: converts from the `.npy` to the `.slcio` file format. Additionally assigns a cell ID to each GenBIB hit.
2. `steer_reco.py`: runs digitization.
3. `steer_BIBtracking.py`: runs reconstruction. *This step is time-intensive, and can take ~20 hours per event.*

Finally, run `compare_tracks.py` to make the plots.

## Running scripts

### Model training

#### Setup

Before running any of the scripts in this repository, you'll want to make a version of `configs.yaml` for your setup. Relevant keywords to set are:

- `PATH_TO_DATA_DIR`: path to wherever your `.slcio` files are on your local machine *these can be downloaded from ])
- `PATH_TO_OUTPUT_DIR`: path where all model training intermediates will be saved to
- `PATH_TO_WANDB_DIR`:  path where all wandb training intermediates will be saved to
- `FEATURE_ORDER`: specify an order for feature training observables. Ensure that all conditioning observables are last. The available observables are: `{0: energy, 1: x, 2: y, 3: z, 4: t, 5: system, 6: side, 7: layer, 8: module, 9: sensor}`
- `NUM_COND_INPUTS`: number of conditioning observables
- `FEATURE_INDICES_DICT`: a dictionary that maps between the ordered indices in `FEATURE_ORDER` to the actual features. If you trin in the `r-phi` basis, you can generally map $x \rightarrow r$ and $y \rightarrow \phi$

#### Scripts

1.   `01_process_data.ipynb`: a notebook mainly for data exploration of the `.slcio` files and to convert them to `.npy` arrays for ML model training. Note that you have the option to save out a different set of observable than `["Edep", "x", "y", "z", "t", "system", "side", "layer", "module", "sensor"]` if you so choose.
2.   `02_generate_samples.py`: the main training script. Here, you'll have the option to choose whether you train the `flow` or `tabddpm` model with the flag `--MODEL`.  We recommend training (`--TRAIN`) and sampling (`--EVAL`) in separate training runs, since the sampling process can take quite a bit of time depending on your machine. 
3.   `03_evaluate_samples.py`: evaluate the quality of the ML models hits by training a BDT to discriminate the full simulation BIB from the GenBIB

For convenience, we also have a notebook `nice_plots.ipynb`, which was used to make all of the plots in the accompanying paper.

### Track fitting

#### Setup

The next set of scripts need to be run on a machine with the muon collider software image available. Details of how to access the image are [here](https://mcd-wiki.web.cern.ch/software/tutorials/fermilab2024/). The scripts in this repository use version 2.11.

First you'll need to place the GenBIB `.npy` files into a folder `<folder_name>`, which will then be the first argument of the `run_pipeline` method in `osg/run_osg.sh`. Within the folder, the files should be named after the corresponding collection e.g `OuterTrackerBarrelCollection.npy`. 

#### Scripts

All of the scripts are in the `osg` folder on this repository. We have compiled them into a helpful bash script `osg/run_osg.sh`, which runs 3 sets of commands:

1. `numpy_to_lcio.py`: converts from the `.npy` to the `.slcio` file format. Additionally assigns a cell ID to each GenBIB hit.
2. `steer_reco.py`: runs digitization
3. `steer_BIBtracking`: runs reconstruction. *This step is time-intensive, and can ~20 hours per event.*

Finally, run `compare_tracks.py` to make the plots.

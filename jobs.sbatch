#!/bin/bash
#SBATCH --account=pi-rmastand    # <-- change to an allowed account
#SBATCH --job-name=muc
#SBATCH --partition=gpu    # <-- change to an allowed partition
#SBATCH --gres=gpu:1
#SBATCH --nodelist=midway3-0286
#SBATCH --time=00:5:00
#SBATCH --array=2

#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmastand@rcc.uchicago.edu  # Where to send email

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/rmastand/muon_collider_env
cd /home/rmastand/muon_collider

commands=(
  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --COLLECTION_LIST OuterTrackerBarrelCollection     \
    --TRAIN  \
    --EVAL       \
    --NAME OTBC_cond4    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BINS 48           \
    --DEGREE 4      \
    --POLYNOMIALS 5    \
    --BASIS rphi       \
    --OVERSAMPLE 2  \
     --TRAINING_FRAC 0.05"

    "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --COLLECTION_LIST InnerTrackerBarrelCollection     \
    --TRAIN  \
    --EVAL       \
    --NAME ITBC_cond4    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BINS 48           \
    --DEGREE 4      \
    --POLYNOMIALS 5    \
    --BASIS rphi       \
    --OVERSAMPLE 2  \
    --TRAINING_FRAC 0.05"




    "python 02_generate_samples.py \
    --MODEL tabddpm  \
    --NUM_EPOCHS 1   \
    --COLLECTION_LIST OuterTrackerBarrelCollection   \
    --TRAIN  \
    --EVAL  \
    --NAME OTBC_cond4 \
    --NUM_COND_INPUTS 4 \
    --BASIS rphi   \
    --OVERSAMPLE 2   \
    --TRAINING_FRAC 0.01 "

  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}


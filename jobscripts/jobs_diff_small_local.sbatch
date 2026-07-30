#!/bin/bash
#SBATCH --account=pi-rmastand    # <-- change to an allowed account
#SBATCH --job-name=diff_local_small
#SBATCH --partition=gpu    # <-- change to an allowed partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint=rtx6000
#SBATCH --time=36:00:00
#SBATCH --array=5

#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmastand@rcc.uchicago.edu  # Where to send email

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/rmastand/muon_collider_env
cd /home/rmastand/muon_collider

commands=(

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST OuterTrackerBarrelCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 0.002        \
    --NAME diff_OTBC_small_cond4_local \
    --BASIS local_phi   \
    --STEPS 200000 \
    --D_LAYERS 2048,2048,2048,2048,2048 \
    --DIM_T 1024 \
    --NUM_TIMESTEPS 1000 "

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST InnerTrackerBarrelCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 0.002        \
    --NAME diff_ITBC_small_cond4_local \
    --BASIS local_phi   \
    --STEPS 200000 \
    --D_LAYERS 2048,2048,2048,2048,2048 \
    --DIM_T 1024 \
    --NUM_TIMESTEPS 1000 "

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST VertexBarrelCollection   \
    --TRAIN  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 0.002        \
    --NAME diff_VBC_small_cond4_local \
    --BASIS local_phi   \
    --STEPS 200000 \
    --D_LAYERS 2048,2048,2048,2048,2048 \
    --DIM_T 1024 \
    --NUM_TIMESTEPS 1000 "

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST OuterTrackerEndcapCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 0.002        \
    --NAME diff_OTEC_small_cond4_local \
    --BASIS local_phi   \
    --STEPS 200000 \
    --D_LAYERS 2048,2048,2048,2048,2048 \
    --DIM_T 1024 \
    --NUM_TIMESTEPS 1000 "

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST InnerTrackerEndcapCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 0.002        \
    --NAME diff_ITEC_small_cond4_local \
    --BASIS local_phi   \
    --STEPS 200000 \
    --D_LAYERS 2048,2048,2048,2048,2048 \
    --DIM_T 1024 \
    --NUM_TIMESTEPS 1000 "

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST VertexEndcapCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 0.002        \
    --NAME diff_VEC_small_cond4_local \
    --BASIS local_phi   \
    --STEPS 200000 \
    --D_LAYERS 2048,2048,2048,2048,2048 \
    --DIM_T 1024 \
    --NUM_TIMESTEPS 1000 "


  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}


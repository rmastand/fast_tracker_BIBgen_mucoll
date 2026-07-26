#!/bin/bash
#SBATCH --account=pi-rmastand    # <-- change to an allowed account
#SBATCH --job-name=muc
#SBATCH --partition=gpu    # <-- change to an allowed partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --array=0

#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmastand@rcc.uchicago.edu  # Where to send email

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/rmastand/muon_collider_env
cd /home/rmastand/muon_collider

commands=(
  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST OuterTrackerBarrelCollection     \
    --EVAL       \
    --NAME flow_OTBC_cond4_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BASIS local_phi       \
    --OVERSAMPLE 2 "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST OuterTrackerEndcapCollection     \
    --TRAIN  \
    --EVAL       \
    --NAME flow_OTEC_cond4_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BASIS local_phi       \
    --OVERSAMPLE 2 "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST InnerTrackerBarrelCollection     \
    --TRAIN  \
    --EVAL       \
    --NAME flow_ITBC_cond4_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BASIS local_phi       \
    --OVERSAMPLE 2 "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST InnerTrackerEndcapCollection     \
    --TRAIN  \
    --EVAL       \
    --NAME flow_ITEC_cond4_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BASIS local_phi       \
    --OVERSAMPLE 2 "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST VertexBarrelCollection     \
    --TRAIN  \
    --EVAL       \
    --NAME flow_VBC_cond4_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BASIS local_phi       \
    --OVERSAMPLE 2 "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST VertexEndcapCollection     \
    --TRAIN  \
    --EVAL       \
    --NAME flow_VEC_cond4_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --NUM_COND_INPUTS 4      \
    --BASIS local_phi       \
    --OVERSAMPLE 2 "






  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}


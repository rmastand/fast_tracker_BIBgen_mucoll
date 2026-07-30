#!/bin/bash
#SBATCH --account=pi-rmastand    # <-- change to an allowed account
#SBATCH --job-name=flow_NCSF_global
#SBATCH --partition=gpu    # <-- change to an allowed partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint=rtx6000
#SBATCH --time=36:00:00
#SBATCH --array=7


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
    --TRAIN  \
    --NAME flow_OTBC_cond4   \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS rphi      "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST OuterTrackerEndcapCollection     \
    --TRAIN  \
    --NAME flow_OTEC_cond4    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS rphi      "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST InnerTrackerBarrelCollection     \
    --TRAIN  \
    --NAME flow_ITBC_cond4    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS rphi      "


  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST InnerTrackerEndcapCollection     \
    --TRAIN  \
    --NAME flow_ITEC_cond4    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS rphi       "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST VertexBarrelCollection     \
    --TRAIN  \
    --NAME flow_VBC_cond4    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS rphi       "

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 2000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 48           \
    --COLLECTION_LIST VertexEndcapCollection     \
    --TRAIN  \
    --NAME flow_VEC_cond4    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS rphi     "






  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}


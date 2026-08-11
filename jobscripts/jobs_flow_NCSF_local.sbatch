#!/bin/bash
#SBATCH --account=pi-rmastand    # <-- change to an allowed account
#SBATCH --job-name=flow_NCSF_local
#SBATCH --partition=gpu    # <-- change to an allowed partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint=rtx6000
#SBATCH --time=36:00:00
#SBATCH --array=0-3


#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmastand@rcc.uchicago.edu  # Where to send email

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/rmastand/muon_collider_env
cd /home/rmastand/muon_collider

commands=(

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 8           \
    --COLLECTION_LIST OuterTrackerBarrelCollection     \
    --EVAL \
    --NAME flow_OTBC_bins8_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS local_phi     \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

    "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 8           \
    --COLLECTION_LIST InnerTrackerBarrelCollection     \
    --EVAL \
    --NAME flow_ITBC_bins8_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS local_phi     \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"


  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 8           \
    --COLLECTION_LIST VertexBarrelCollection     \
    --EVAL \
    --NAME flow_VBC_bins8_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS local_phi     \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 8           \
    --COLLECTION_LIST OuterTrackerEndcapCollection     \
    --EVAL \
    --NAME flow_OTEC_bins8_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS local_phi      \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 8           \
    --COLLECTION_LIST InnerTrackerEndcapCollection     \
    --EVAL \
    --NAME flow_ITEC_bins8_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS local_phi     \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

  "python 02_generate_samples.py  \
    --ZUKO_ID NCSF   \
    --NUM_EPOCHS 1000      \
    --TRANSFORMS 5        \
    --HIDDEN_FEATURES 128,128,128     \
    --BINS 8           \
    --COLLECTION_LIST VertexEndcapCollection     \
    --EVAL \
    --NAME flow_VEC_bins8_local    \
    --BATCH_SIZE 2048        \
    --LEARNING_RATE 0.0001        \
    --BASIS local_phi       \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"
 

 


  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}


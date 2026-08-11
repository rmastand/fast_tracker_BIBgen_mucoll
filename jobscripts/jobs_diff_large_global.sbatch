#!/bin/bash
#SBATCH --account=pi-rmastand    # <-- change to an allowed account
#SBATCH --job-name=diff_global
#SBATCH --partition=gpu    # <-- change to an allowed partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint=rtx6000
#SBATCH --time=00:15:00
#SBATCH --array=0

#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmastand@rcc.uchicago.edu  # Where to send email

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/rmastand/muon_collider_env
cd /home/rmastand/muon_collider

commands=(

  "python -u 02_generate_samples.py     --MODEL tabddpm      --COLLECTION_LIST OuterTrackerBarrelCollection      --EVAL      --BATCH_SIZE 4096            --LEARNING_RATE 5e-4            --NAME diff_OTBC_local     --BASIS local_phi --STEPS 2000 "

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST InnerTrackerBarrelCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 5e-4        \
    --NAME diff_ITBC_cond4 \
    --BASIS rphi \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST VertexBarrelCollection   \
    --EVAL    \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 5e-4        \
    --NAME diff_VBC_cond4 \
    --BASIS rphi \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST OuterTrackerEndcapCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 5e-4        \
    --NAME diff_OTEC_cond4 \
    --BASIS rphi \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST InnerTrackerEndcapCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 5e-4        \
    --NAME diff_ITEC_cond4 \
    --BASIS rphi  \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"

  "python -u 02_generate_samples.py \
    --MODEL tabddpm  \
    --COLLECTION_LIST VertexEndcapCollection   \
    --EVAL  \
    --BATCH_SIZE 4096        \
    --LEARNING_RATE 5e-4        \
    --NAME diff_VEC_cond4 \
    --BASIS rphi \
    --CONTEXT_FILE  /scratch/midway3/rmastand/muon_collider/nuGun_pT_0_50/reco_h5/nuGun_pT_0_50_reco_1_contexts.npy"


  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}


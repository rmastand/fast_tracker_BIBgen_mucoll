"""
Compile sampling_intermediates from a timed-out 02_generate_samples tabddpm job
into the final generated_samples.npy output.

Usage:
    python compile_sampling_intermediates.py <intermediates_dir> <output_path>

Example:
    python compile_sampling_intermediates.py \
        /scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_OTBC_local/sampling_intermediates_17 \
        /scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_OTBC_local/generated_samples_from_context_17.npy
"""
import sys
import re
import numpy as np
from pathlib import Path

intermediates_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])

index_files = sorted(intermediates_dir.glob("round_*_indices.npy"))
if not index_files:
    raise FileNotFoundError(f"No round_*_indices.npy files found in {intermediates_dir}")

print(f"Found {len(index_files)} rounds.")

# Determine output array shape from one samples file
first_rnd = re.search(r'round_(\d+)_indices', index_files[0].name).group(1)
sample_shape = np.load(intermediates_dir / f"round_{first_rnd}_samples.npy").shape[1]

# Find total number of slots from the max index seen
max_idx = max(int(np.load(f).max()) for f in index_files)
num_samples_total = max_idx + 1
print(f"Total slots: {num_samples_total}, features per row: {sample_shape}")

output_samples = np.empty((num_samples_total, sample_shape), dtype=np.float32)
filled = np.zeros(num_samples_total, dtype=bool)

for idx_path in index_files:
    rnd = re.search(r'round_(\d+)_indices', idx_path.name).group(1)
    samp_path = intermediates_dir / f"round_{rnd}_samples.npy"
    r_idx = np.load(idx_path)
    r_samp = np.load(samp_path)
    output_samples[r_idx] = r_samp
    filled[r_idx] = True

n_unfilled = int((~filled).sum())
if n_unfilled:
    print(f"WARNING: {n_unfilled}/{num_samples_total} slots unfilled — saving only filled rows.")

final = output_samples[filled]
np.save(output_path, final)
print(f"Saved {len(final)} samples to {output_path}")

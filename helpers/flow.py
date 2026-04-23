import numpy as np
import torch

def sample_from_flow(flow, N=None, x_context=None):
    flow.eval()
    with torch.no_grad():
        if x_context is not None:
            samples = flow(x_context).sample().detach().cpu().numpy()
            samples = np.hstack([
                        samples,
                        x_context.cpu().numpy()])
        else:
            samples = flow().sample((N,)).detach().cpu().numpy()

        return samples

def build_z_lookup(all_data, side_idx, layer_idx, sensor_idx, z_idx):
    sides = np.asarray(all_data[:, side_idx]).reshape(-1).astype(int)
    layers = np.asarray(all_data[:, layer_idx]).reshape(-1).astype(int)
    sensors = np.asarray(all_data[:, sensor_idx]).reshape(-1).astype(int)
    zs = np.asarray(all_data[:, z_idx]).reshape(-1)

    lookup = {}

    for (s, l, sens) in zip(sides, layers, sensors):
        key = (s, l, sens)
        if key not in lookup:
            lookup[key] = []

    # fill (faster than repeated masking)
    for s, l, sens, z in zip(sides, layers, sensors, zs):
        lookup[(s, l, sens)].append(z)

    # convert lists → numpy arrays
    for key in lookup:
        lookup[key] = np.array(lookup[key])


    return lookup


def snap_z_to_detector(z_vals, sides, layers, sensors, lookup):
    # ensure clean 1D integer keys
    sides = np.asarray(sides).reshape(-1).astype(int)
    layers = np.asarray(layers).reshape(-1).astype(int)
    sensors = np.asarray(sensors).reshape(-1).astype(int)
    z_vals = np.asarray(z_vals).reshape(-1)

    z_new = np.empty(len(z_vals), dtype=np.float64)


    unique_keys = set(zip(sides, layers, sensors))

    for key in unique_keys:
        if key not in lookup:
            # fallback (don’t crash)
            mask = (sides == key[0]) & (layers == key[1]) & (sensors == key[2])
            z_new[mask] = z_vals[mask]
            continue

        mask = (sides == key[0]) & (layers == key[1]) & (sensors == key[2])
        idxs = np.where(mask)[0]

        z_choices = lookup[key]   # <-- guaranteed numpy array now
        rand_idxs = np.random.randint(0, len(z_choices), size=len(idxs))

        z_new[idxs] = z_choices[rand_idxs]


    return z_new
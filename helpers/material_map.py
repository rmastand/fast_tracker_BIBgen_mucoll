import numpy as np


def build_material_map_3D(data_dir, features, bins=(300, 300, 300)):
    """
    Build a 3D occupancy map in (r, phi, z)

    Args:
        data: (N, D) array with (r, phi) or (x,y,z)
        bins: number of bins in (r, phi, z)

    Returns:
        H_mask: (Nr, Nphi, Nz) boolean occupancy
        edges: tuple of bin edges
    """
    material_map = {}
    
    for col_name in data_dir.keys():
        
        # Convert to cylindrical if needed
        if features == "rphi":
            r = data_dir[col_name][:, 1]
            phi = data_dir[col_name][:, 2]
            z = data_dir[col_name][:, 3]
        else:
            x, y, z = data_dir[col_name][:,1], data_dir[col_name][:,2], data_dir[col_name][:,3]
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)

            
    
        # Histogram
        H, edges = np.histogramdd(
            np.stack([r, phi, z], axis=1),
            bins=bins
        )
        r_edges, phi_edges, z_edges = edges
    
        # Convert to occupancy mask
        H_mask = H > 0

        material_map[col_name] = H_mask, r_edges, phi_edges, z_edges

    return material_map
        
    


def apply_material_map_3D(samples_dir, material_map, col_name, features):
    """
    Apply 3D material map mask to samples
    """

    if features == "rphi":
        r = samples_dir[col_name][:, 1]
        phi = samples_dir[col_name][:, 2]
        z = samples_dir[col_name][:, 3] 
    else:
        x, y, z = samples_dir[col_name][:,1], samples_dir[col_name][:,2], samples_dir[col_name][:,3]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)


    H_mask, r_edges, phi_edges, z_edges = material_map[col_name]

    # Bin indices
    r_idx = np.digitize(r, r_edges) - 1
    phi_idx = np.digitize(phi, phi_edges) - 1
    z_idx = np.digitize(z, z_edges) - 1

    # Valid indices
    valid = (
        (r_idx >= 0) & (r_idx < H_mask.shape[0]) &
        (phi_idx >= 0) & (phi_idx < H_mask.shape[1]) &
        (z_idx >= 0) & (z_idx < H_mask.shape[2])
    )

    mask = np.zeros(len(samples_dir[col_name]), dtype=bool)

    mask[valid] = H_mask[
        r_idx[valid],
        phi_idx[valid],
        z_idx[valid]
    ]

    return mask








def build_material_map_5D(data_dir, features, N_BINS=300):
    """
    Same as above but with layer, system 
    """
    material_map = {}
    
    for col_name in data_dir.keys():
        
        # Convert to cylindrical if needed
        if features == "rphi":
            r = data_dir[col_name][:, 1]
            phi = data_dir[col_name][:, 2]
        else:
            x, y = data_dir[col_name][:,1], data_dir[col_name][:,2]
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)

        z = data_dir[col_name][:, 3]
        side = data_dir[col_name][:, 5]
        layer = data_dir[col_name][:, 6]

        bins_r = np.linspace(0.99*r.min(), 1.01*r.max(), N_BINS)
        bins_phi = np.linspace(1.01*phi.min(), 1.01*phi.max(), N_BINS)
        bins_z = np.linspace(1.01*z.min(), 1.01*z.max(), N_BINS)
        bins_side = np.arange(-1.5, 2.5, 1)
        bins_layer = np.arange(layer.min()-0.5, layer.max()+1.5, 1)

            
        # Histogram
        H, edges = np.histogramdd(
            np.stack([r, phi, z, side, layer], axis=1),
            bins=[bins_r, bins_phi, bins_z, bins_side, bins_layer]
        )
        r_edges, phi_edges, z_edges, side_edges, layer_edges = edges

        # Convert to occupancy mask
        H_mask = H > 0

        material_map[col_name] = H_mask, r_edges, phi_edges, z_edges, side_edges, layer_edges

    return material_map
        
    


    

def apply_material_map_5D(samples_dir, material_map, col_name, features):
    """
    Apply 5D material map mask to samples
    """

    if features == "rphi":
        r = samples_dir[col_name][:, 1]
        phi = samples_dir[col_name][:, 2]
       
    else:
        x, y = samples_dir[col_name][:,1], samples_dir[col_name][:,2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

    z = samples_dir[col_name][:, 3] 
    side = samples_dir[col_name][:, 5]
    layer = samples_dir[col_name][:, 6]


    H_mask, r_edges, phi_edges, z_edges, side_edges, layer_edges = material_map[col_name]

    # Bin indices
    r_idx = np.digitize(r, r_edges) - 1
    phi_idx = np.digitize(phi, phi_edges) - 1
    z_idx = np.digitize(z, z_edges) - 1
    side_idx = np.digitize(side, side_edges) - 1
    layer_idx = np.digitize(layer, layer_edges) - 1

    # Valid indices
    valid = (
        (r_idx >= 0) & (r_idx < H_mask.shape[0]) &
        (phi_idx >= 0) & (phi_idx < H_mask.shape[1]) &
        (z_idx >= 0) & (z_idx < H_mask.shape[2]) &
        (side_idx >= 0) & (side_idx < H_mask.shape[3]) &
        (layer_idx >= 0) & (layer_idx < H_mask.shape[4]) 
    )

    mask = np.zeros(len(samples_dir[col_name]), dtype=bool)

    mask[valid] = H_mask[
        r_idx[valid],
        phi_idx[valid],
        z_idx[valid],
        side_idx[valid],
        layer_idx[valid]
    ]

    return mask
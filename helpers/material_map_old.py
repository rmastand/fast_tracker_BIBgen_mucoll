

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





def build_material_map_2Dphi(data_dir, master_feature_indices_dict, N_BINS=300):

    
    material_map = {}

    for col_name in data_dir.keys():

        if "Barrel" in col_name:
            tangent_coord = data_dir[col_name][:, master_feature_indices_dict[col_name]["r"]]
            N_BINS = 800
        elif "Endcap" in col_name: 
            tangent_coord = data_dir[col_name][:, master_feature_indices_dict[col_name]["r"]]
            N_BINS = 320
            
        phi = data_dir[col_name][:, master_feature_indices_dict[col_name]["phi"]]
        


        bins_coord = np.linspace(0.99*tangent_coord.min(), 1.01*tangent_coord.max(), N_BINS)
        bins_phi = np.linspace(1.01*phi.min(), 1.01*phi.max(), N_BINS)

        H, tangent_coord_edges, phi_edges = np.histogram2d(tangent_coord, phi, bins=[bins_coord, bins_phi])

        H_mask = H > 0

        material_map[col_name] = (H_mask, tangent_coord_edges, phi_edges)

    return material_map



def apply_material_map_hybrid(samples_dir, material_map, col_name, master_feature_indices_dict):

    data = samples_dir[col_name]

    if "Barrel" in col_name:
        tangent_coord = data[:, master_feature_indices_dict[col_name]["r"]]
        layer_coord = data[:, master_feature_indices_dict[col_name]["r"]]
        geom = r_side_layer_map_barrel[col_name]
    elif "Endcap" in col_name: 
        tangent_coord = data[:, master_feature_indices_dict[col_name]["r"]] 
        layer_coord = data[:, master_feature_indices_dict[col_name]["z"]]
        geom = z_side_layer_map_endcaps[col_name]

    phi = data[:, master_feature_indices_dict[col_name]["phi"]]
    side = data[:, master_feature_indices_dict[col_name]["side"]].astype(int)
    layer_id = data[:, master_feature_indices_dict[col_name]["layer"]].astype(int)

    # --- r, phi mask ---
    H_mask, tangent_coord_edges, phi_edges = material_map[col_name]

    tangent_coord_idx = np.digitize(tangent_coord, tangent_coord_edges) - 1
    phi_idx = np.digitize(phi, phi_edges) - 1

    valid_tangent_coord_phi = (
        (tangent_coord_idx >= 0) & (tangent_coord_idx < H_mask.shape[0]) &
        (phi_idx >= 0) & (phi_idx < H_mask.shape[1])
    )

    mask_tangent_coord_phi = np.zeros(len(data), dtype=bool)
    mask_tangent_coord_phi[valid_tangent_coord_phi] = H_mask[tangent_coord_idx[valid_tangent_coord_phi], phi_idx[valid_tangent_coord_phi]]

 
    mask_geom = np.zeros(len(data), dtype=bool)

    for (s, l), layer_coord_dict in geom.items():
        idx = (side == s) & (layer_id == l)

        if not np.any(idx):
            continue

        layer_coord_vals = layer_coord[idx]

        layer_coord_mask_local = np.zeros_like(layer_coord_vals, dtype=bool)

        for start, stop in zip(layer_coord_dict["starts"], layer_coord_dict["stops"]):
            layer_coord_mask_local |= (layer_coord_vals >= start) & (layer_coord_vals <= stop)

        mask_geom[idx] = layer_coord_mask_local

    # --- final mask ---
    return mask_tangent_coord_phi & mask_geom



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


z_side_layer_map_endcaps = {'InnerTrackerEndcapCollection': {(1, 0): {'starts': [522.849143562149, 528.8496556051206], 'stops': [522.8533456013948, 528.8538576443664]}, (1, 1): {'starts': [803.1461424545209, 809.1466505538285], 'stops': [803.1503444910051, 809.1508525903126]}, (1, 2): {'starts': [1091.8491427589308, 1097.8496554190353], 'stops': [1091.8533447986088, 1097.8538574587133]}, (1, 3): {'starts': [1372.1461428038197, 1378.1466468450215], 'stops': [1372.150344837462, 1378.1508488786637]}, (1, 4): {'starts': [1659.849143382624, 1665.8496551612313], 'stops': [1659.8533454216847, 1665.853857200292]}, (1, 5): {'starts': [1941.14614220122, 1947.1466522540668], 'stops': [1941.1503442390722, 1947.150854291919]}, (1, 6): {'starts': [2188.8491463031573, 2194.849654804879], 'stops': [2188.8533483399233, 2194.8538568416448]}, (-1, 0): {'starts': [-528.8538575003843, -522.8533441678617], 'stops': [-528.8496554602355, -522.8491421277129]}, (-1, 1): {'starts': [-809.1508543740399, -803.1503441588799], 'stops': [-809.1466523360741, -803.1461421209141]}, (-1, 2): {'starts': [-1097.8538571301856, -1091.8533453594218], 'stops': [-1097.8496550911304, -1091.8491433203667]}, (-1, 3): {'starts': [-1378.1508541816436, -1372.1503459821215], 'stops': [-1378.1466521450893, -1372.1461439455672]}, (-1, 4): {'starts': [-1665.8538513466156, -1659.8533491809549], 'stops': [-1665.8496493142868, -1659.8491471486261]}, (-1, 5): {'starts': [-1947.1508578598198, -1941.1503491779981], 'stops': [-1947.1466558229276, -1941.146147141106]}, (-1, 6): {'starts': [-2194.853854018727, -2188.8533447803366], 'stops': [-2194.8496519814453, -2188.849142743055]}}, 'OuterTrackerEndcapCollection': {(1, 0): {'starts': [1305.1461421633815, 1311.1466553002076], 'stops': [1305.1503442033934, 1311.1508573402195]}, (1, 1): {'starts': [1615.849141970229, 1621.8496537024448], 'stops': [1615.8533440092572, 1621.853855741473]}, (1, 2): {'starts': [1878.1461429805634, 1884.1466555550721], 'stops': [1878.1503450201815, 1884.1508575946903]}, (1, 3): {'starts': [2188.849143000882, 2194.849655625588], 'stops': [2188.8533450405357, 2194.853857665242]}, (-1, 0): {'starts': [-1311.1508565225504, -1305.1503443189094], 'stops': [-1311.1466544831922, -1305.1461422795512]}, (-1, 1): {'starts': [-1621.8538551980234, -1615.8533445284986], 'stops': [-1621.8496531597395, -1615.8491424902147]}, (-1, 2): {'starts': [-1884.1508569963762, -1878.1503446670397], 'stops': [-1884.14665495693, -1878.1461426275935]}, (-1, 3): {'starts': [-2194.8538577922936, -2188.8533446672172], 'stops': [-2194.8496557522903, -2188.849142627214]}}, 'VertexEndcapCollection': {(1, 0): {'starts': [80.13889753902934], 'stops': [80.14110237939136]}, (1, 1): {'starts': [84.188897861894], 'stops': [84.19110270146345]}, (1, 2): {'starts': [120.13889763560879], 'stops': [120.14110247612658]}, (1, 3): {'starts': [124.18889759575274], 'stops': [124.19110243659355]}, (1, 4): {'starts': [200.13889748444515], 'stops': [200.14110232480758]}, (1, 5): {'starts': [204.18889767615738], 'stops': [204.19110251665663]}, (1, 6): {'starts': [280.13889765423824], 'stops': [280.1411024943865]}, (1, 7): {'starts': [284.18889776666776], 'stops': [284.1911026063793]}, (-1, 0): {'starts': [-80.14110237190039], 'stops': [-80.13889753183304]}, (-1, 1): {'starts': [-84.19110240027032], 'stops': [-84.1888975596242]}, (-1, 2): {'starts': [-120.1411023063737], 'stops': [-120.13889746609716]}, (-1, 3): {'starts': [-124.19110254374714], 'stops': [-124.1888977039106]}, (-1, 4): {'starts': [-200.14110244532807], 'stops': [-200.1388976055517]}, (-1, 5): {'starts': [-204.1911026621207], 'stops': [-204.18889782292106]}, (-1, 6): {'starts': [-280.1411018246637], 'stops': [-280.1388969864688]}, (-1, 7): {'starts': [-284.19110222185054], 'stops': [-284.18889738388157]}}}


def build_material_map_rphi(data_dir, features, N_BINS=300):
    material_map = {}

    for col_name in data_dir.keys():

        if features == "rphi":
            r = data_dir[col_name][:, 1]
            phi = data_dir[col_name][:, 2]
        else:
            x, y = data_dir[col_name][:,1], data_dir[col_name][:,2]
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)

        bins_r = np.linspace(0.99*r.min(), 1.01*r.max(), N_BINS)
        bins_phi = np.linspace(1.01*phi.min(), 1.01*phi.max(), N_BINS)

        H, r_edges, phi_edges = np.histogram2d(r, phi, bins=[bins_r, bins_phi])

        H_mask = H > 0

        material_map[col_name] = (H_mask, r_edges, phi_edges)

    return material_map

    


def apply_material_map_hybrid(samples_dir, material_map, col_name, features):

    data = samples_dir[col_name]

    if features == "rphi":
        r = data[:, 1]
        phi = data[:, 2]
    else:
        x, y = data[:,1], data[:,2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

    z = data[:, 3]
    side = data[:, 5].astype(int)
    layer = data[:, 6].astype(int)

    # --- r, phi mask ---
    H_mask, r_edges, phi_edges = material_map[col_name]

    r_idx = np.digitize(r, r_edges) - 1
    phi_idx = np.digitize(phi, phi_edges) - 1

    valid_rphi = (
        (r_idx >= 0) & (r_idx < H_mask.shape[0]) &
        (phi_idx >= 0) & (phi_idx < H_mask.shape[1])
    )

    mask_rphi = np.zeros(len(data), dtype=bool)
    mask_rphi[valid_rphi] = H_mask[r_idx[valid_rphi], phi_idx[valid_rphi]]

    # --- geometry mask (z, side, layer) ---
    geom = z_side_layer_map_endcaps[col_name]

    mask_geom = np.zeros(len(data), dtype=bool)

    for (s, l), z_dict in geom.items():
        idx = (side == s) & (layer == l)

        if not np.any(idx):
            continue

        z_vals = z[idx]

        z_mask_local = np.zeros_like(z_vals, dtype=bool)

        for start, stop in zip(z_dict["starts"], z_dict["stops"]):
            z_mask_local |= (z_vals >= start) & (z_vals <= stop)

        mask_geom[idx] = z_mask_local

    # --- final mask ---
    return mask_rphi & mask_geom
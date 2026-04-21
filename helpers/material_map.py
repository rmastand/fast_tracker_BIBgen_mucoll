import numpy as np

####################################################################################################
#
# EXPLICITLY BUILDING THE MODULES FROM THE MAIA XML FILES
#
####################################################################################################

####################################################################################################
#
# BARREL
#
####################################################################################################
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
MODULE_HALF_LENGTH  = 15.05   # mm — tangential half-length of all barrel modules
HALF_SENSITIVE      = 0.050   # mm — half of 0.100mm sensitive Si layer

# ── Inner Tracker Barrel ──────────────────────────────────────────────────────
# rc = sensitive layer radius (DD4hep places sensor exactly at rc)
# nphi_half * 2 = total modules per layer (staggered double ring)
# dr = radial stagger between inner and outer ring

INNER_LAYERS = [
    {'rc': 127.0, 'nphi_half': 14, 'dr': 2.5},
    {'rc': 340.0, 'nphi_half': 38, 'dr': 2.5},
    {'rc': 554.0, 'nphi_half': 62, 'dr': 2.5},
]

def buildInnerTrackerBarrelModules():
    modules = []
    for layer in INNER_LAYERS:
        rc, nphi_half, dr = layer['rc'], layer['nphi_half'], layer['dr']
        nphi = nphi_half * 2
        for iphi in range(nphi):
            phi      = 2 * np.pi * iphi / nphi
            r_sensor = rc if iphi % 2 == 0 else rc + dr
            cx = r_sensor * np.cos(phi)
            cy = r_sensor * np.sin(phi)
            tx, ty = -np.sin(phi), np.cos(phi)
            modules.append({
                'x0': cx - MODULE_HALF_LENGTH * tx,
                'y0': cy - MODULE_HALF_LENGTH * ty,
                'x1': cx + MODULE_HALF_LENGTH * tx,
                'y1': cy + MODULE_HALF_LENGTH * ty,
            })
    return modules

# ── Outer Tracker Barrel ──────────────────────────────────────────────────────
# rc = module envelope centre (sensor offset must be computed from stack)
# drp/drm = radial stagger for even/odd phi modules
# sensor_offset = distance from module inner face to sensor centre
#   layers 0,1 use ModuleIn  (sensor near inner face, offset = 0.8015mm)
#   layer  2   uses ModuleOut (sensor near outer face, offset = 4.500mm)

OUTER_HALF_STACK = 5.303 / 2   # mm — half of total module stack thickness

OUTER_LAYERS = [
    {'rc': 819.0,  'nphi': 184, 'drp': 0.0, 'drm': 5.5, 'sensor_offset': 0.8015},
    {'rc': 1153.0, 'nphi': 256, 'drp': 0.0, 'drm': 5.5, 'sensor_offset': 0.8015},
    {'rc': 1486.0, 'nphi': 328, 'drp': 0.0, 'drm': 5.5, 'sensor_offset': 4.500},
]

def buildOuterTrackerBarrelModules():
    modules = []
    for layer in OUTER_LAYERS:
        rc   = layer['rc']
        nphi = layer['nphi']
        drp  = layer['drp']
        drm  = layer['drm']
        sensor_offset = layer['sensor_offset']
        for iphi in range(nphi):
            phi       = 2 * np.pi * iphi / nphi
            r_nominal = (rc - drp) if iphi % 2 == 0 else (rc + drm)
            r_sensor  = r_nominal - OUTER_HALF_STACK + sensor_offset
            cx = r_sensor * np.cos(phi)
            cy = r_sensor * np.sin(phi)
            tx, ty = -np.sin(phi), np.cos(phi)
            modules.append({
                'x0': cx - MODULE_HALF_LENGTH * tx,
                'y0': cy - MODULE_HALF_LENGTH * ty,
                'x1': cx + MODULE_HALF_LENGTH * tx,
                'y1': cy + MODULE_HALF_LENGTH * ty,
            })
    return modules

# ── Vertex Barrel ─────────────────────────────────────────────────────────────
# Flat ladder staves (ZSegmentedPlanarTracker)
# r = inner face of support; sensitive layer at r + SUPPORT_THICKNESS
# offset = tangential shift of stave centre from radial direction
# only innermost layer has a double-layer (inner + outer sensitive surface)

VERTEX_SUPPORT_THICKNESS  = 0.140  # mm
VERTEX_SENSITIVE_THICKNESS = 0.050  # mm
VERTEX_DOUBLELAYER_GAP     = 2.0    # mm

VERTEX_LAYERS = [
    {'r':  30.0, 'nstaves': 16, 'width': 13.0, 'offset': 2.0, 'double': True},
    {'r':  51.0, 'nstaves': 15, 'width': 23.0, 'offset': 2.0, 'double': False},
    {'r':  74.0, 'nstaves': 21, 'width': 24.0, 'offset': 2.0, 'double': False},
    {'r': 102.0, 'nstaves': 29, 'width': 24.0, 'offset': 2.0, 'double': False},
]

def buildVertexBarrelModules():
    modules = []
    for layer in VERTEX_LAYERS:
        r, nstaves, width, offset = layer['r'], layer['nstaves'], layer['width'], layer['offset']
        r_inner = r + VERTEX_SUPPORT_THICKNESS
        r_outer = r_inner + VERTEX_SENSITIVE_THICKNESS + VERTEX_DOUBLELAYER_GAP
        radii   = [r_inner, r_outer] if layer['double'] else [r_inner]
        half_w  = width / 2.0
        for r_sens in radii:
            for i in range(nstaves):
                phi    = 2 * np.pi * i / nstaves
                tx, ty = -np.sin(phi), np.cos(phi)
                cx = r_sens * np.cos(phi) + offset * tx
                cy = r_sens * np.sin(phi) + offset * ty
                modules.append({
                    'x0': cx - half_w * tx,
                    'y0': cy - half_w * ty,
                    'x1': cx + half_w * tx,
                    'y1': cy + half_w * ty,
                })
    return modules


# ── Mask function ─────────────────────────────────────────────────────────────
def make_barrel_mask(x, y, modules, corridor_width=HALF_SENSITIVE):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.zeros(len(x), dtype=bool)
    for m in modules:
        dx, dy = m['x1'] - m['x0'], m['y1'] - m['y0']
        len_sq = dx*dx + dy*dy
        t  = np.clip(((x - m['x0'])*dx + (y - m['y0'])*dy) / len_sq, 0, 1)
        px = m['x0'] + t*dx
        py = m['y0'] + t*dy
        mask |= (x - px)**2 + (y - py)**2 <= corridor_width**2
    return mask


####################################################################################################
#
# ENDCAP
#
####################################################################################################



from matplotlib.path import Path

def make_polygon_vertices(nsides, inradius, rotation_deg=0.0):
    """vertices of regular polygon with inradius (flat-edge radius) = inradius."""
    circumradius = inradius / np.cos(np.pi / nsides)
    angles = np.linspace(0, 2*np.pi, nsides, endpoint=False) + np.deg2rad(rotation_deg)
    return np.stack([circumradius * np.cos(angles), circumradius * np.sin(angles)], axis=1)

def points_in_polygon(x, y, vertices):
    path = Path(np.vstack([vertices, vertices[0]]))
    return path.contains_points(np.stack([x, y], axis=1))

    

def make_polygonal_annulus_mask(x, y, z, disks, nsides=12, rotation_deg=90.0, z_tolerance=2.0, reflect=True):
    """
    Polygonal annulus mask — both inner and outer boundaries are polygons.
    disks: list of {'z', 'rmin', 'rmax'}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # z = np.asarray(z, dtype=float)

    mask = np.zeros(len(x), dtype=bool)

    for disk in disks:
        # z_match = np.abs(z - disk['z']) <= z_tolerance
        # if reflect:
        #     z_match |= np.abs(z + disk['z']) <= z_tolerance

        # if not z_match.any():
        #     continue

        outer_verts = make_polygon_vertices(nsides, disk['rmax'], rotation_deg)
        inner_verts = make_polygon_vertices(nsides, disk['rmin'], rotation_deg)

        in_outer = points_in_polygon(x, y, outer_verts)
        in_inner = points_in_polygon(x, y, inner_verts)

        mask |=  in_outer & ~in_inner

    return mask

# ── Disk definitions ──────────────────────────────────────────────────────────

VERTEX_ENDCAP_DISKS = [
    {'z': 80,   'rmin': 25,  'rmax': 112},
    {'z': 82,   'rmin': 25,  'rmax': 112},
    {'z': 120,  'rmin': 31,  'rmax': 112},
    {'z': 122,  'rmin': 31,  'rmax': 112},
    {'z': 200,  'rmin': 38,  'rmax': 112},
    {'z': 202,  'rmin': 38,  'rmax': 112},
    {'z': 280,  'rmin': 53,  'rmax': 112},
    {'z': 282,  'rmin': 53,  'rmax': 112},
]

INNER_ENDCAP_DISKS = [
    {'z': 524,  'rmin': 95,  'rmax': 405},
    {'z': 808,  'rmin': 147, 'rmax': 555},
    {'z': 1093, 'rmin': 190, 'rmax': 555},
    {'z': 1377, 'rmin': 212, 'rmax': 555},
    {'z': 1661, 'rmin': 237, 'rmax': 555},
    {'z': 1946, 'rmin': 264, 'rmax': 555},
    {'z': 2190, 'rmin': 284, 'rmax': 555},
]

OUTER_ENDCAP_DISKS = [
    {'z': 1310, 'rmin': 617.5, 'rmax': 1430.2},
    {'z': 1617, 'rmin': 617.5, 'rmax': 1430.2},
    {'z': 1883, 'rmin': 617.5, 'rmax': 1430.2},
    {'z': 2190, 'rmin': 617.5, 'rmax': 1430.2},
]




z_side_layer_map_endcaps = {
    'InnerTrackerEndcapCollection': 
    {(1, 0): {'starts': [522.849143562149, 528.8496556051206], 'stops': [522.8533456013948, 528.8538576443664]}, 
     (1, 1): {'starts': [803.1461424545209, 809.1466505538285], 'stops': [803.1503444910051, 809.1508525903126]}, 
     (1, 2): {'starts': [1091.8491427589308, 1097.8496554190353], 'stops': [1091.8533447986088, 1097.8538574587133]}, 
     (1, 3): {'starts': [1372.1461428038197, 1378.1466468450215], 'stops': [1372.150344837462, 1378.1508488786637]}, 
     (1, 4): {'starts': [1659.849143382624, 1665.8496551612313], 'stops': [1659.8533454216847, 1665.853857200292]}, 
     (1, 5): {'starts': [1941.14614220122, 1947.1466522540668], 'stops': [1941.1503442390722, 1947.150854291919]}, 
     (1, 6): {'starts': [2188.8491463031573, 2194.849654804879], 'stops': [2188.8533483399233, 2194.8538568416448]}, 
     (-1, 0): {'starts': [-528.8538575003843, -522.8533441678617], 'stops': [-528.8496554602355, -522.8491421277129]}, 
     (-1, 1): {'starts': [-809.1508543740399, -803.1503441588799], 'stops': [-809.1466523360741, -803.1461421209141]}, 
     (-1, 2): {'starts': [-1097.8538571301856, -1091.8533453594218], 'stops': [-1097.8496550911304, -1091.8491433203667]}, 
     (-1, 3): {'starts': [-1378.1508541816436, -1372.1503459821215], 'stops': [-1378.1466521450893, -1372.1461439455672]}, 
     (-1, 4): {'starts': [-1665.8538513466156, -1659.8533491809549], 'stops': [-1665.8496493142868, -1659.8491471486261]}, 
     (-1, 5): {'starts': [-1947.1508578598198, -1941.1503491779981], 'stops': [-1947.1466558229276, -1941.146147141106]}, 
     (-1, 6): {'starts': [-2194.853854018727, -2188.8533447803366], 'stops': [-2194.8496519814453, -2188.849142743055]}}, 
    
   'OuterTrackerEndcapCollection': {(1, 0): {'starts': [1305.1461421633815, 1311.1466553002076], 'stops': [1305.1503442033934, 1311.1508573402195]}, (1, 1): {'starts': [1615.849141970229, 1621.8496537024448], 'stops': [1615.8533440092572, 1621.853855741473]}, (1, 2): {'starts': [1878.1461429805634, 1884.1466555550721], 'stops': [1878.1503450201815, 1884.1508575946903]}, (1, 3): {'starts': [2188.849143000882, 2194.849655625588], 'stops': [2188.8533450405357, 2194.853857665242]}, (-1, 0): {'starts': [-1311.1508565225504, -1305.1503443189094], 'stops': [-1311.1466544831922, -1305.1461422795512]}, (-1, 1): {'starts': [-1621.8538551980234, -1615.8533445284986], 'stops': [-1621.8496531597395, -1615.8491424902147]}, (-1, 2): {'starts': [-1884.1508569963762, -1878.1503446670397], 'stops': [-1884.14665495693, -1878.1461426275935]}, (-1, 3): {'starts': [-2194.8538577922936, -2188.8533446672172], 'stops': [-2194.8496557522903, -2188.849142627214]}}, 
    
    'VertexEndcapCollection': {(1, 0): {'starts': [80.13889753902934], 'stops': [80.14110237939136]}, (1, 1): {'starts': [84.188897861894], 'stops': [84.19110270146345]}, (1, 2): {'starts': [120.13889763560879], 'stops': [120.14110247612658]}, (1, 3): {'starts': [124.18889759575274], 'stops': [124.19110243659355]}, (1, 4): {'starts': [200.13889748444515], 'stops': [200.14110232480758]}, (1, 5): {'starts': [204.18889767615738], 'stops': [204.19110251665663]}, (1, 6): {'starts': [280.13889765423824], 'stops': [280.1411024943865]}, (1, 7): {'starts': [284.18889776666776], 'stops': [284.1911026063793]}, (-1, 0): {'starts': [-80.14110237190039], 'stops': [-80.13889753183304]}, (-1, 1): {'starts': [-84.19110240027032], 'stops': [-84.1888975596242]}, (-1, 2): {'starts': [-120.1411023063737], 'stops': [-120.13889746609716]}, (-1, 3): {'starts': [-124.19110254374714], 'stops': [-124.1888977039106]}, (-1, 4): {'starts': [-200.14110244532807], 'stops': [-200.1388976055517]}, (-1, 5): {'starts': [-204.1911026621207], 'stops': [-204.18889782292106]}, (-1, 6): {'starts': [-280.1411018246637], 'stops': [-280.1388969864688]}, (-1, 7): {'starts': [-284.19110222185054], 'stops': [-284.18889738388157]}}
}

r_side_layer_map_barrel = {
    'InnerTrackerBarrelCollection': {(0, 0): {'starts': [126.96707025599089, 129.46723334726383], 'stops': [127.85848599777086, 130.34115319617277]}, (0, 1): {'starts': [339.96734508187086, 342.46724292384977], 'stops': [340.30311792909157, 342.7999493523744]}, (0, 2): {'starts': [553.9671055170247, 556.4672189952502], 'stops': [554.1739041118735, 556.6732763764973]}}, 
    'OuterTrackerBarrelCollection': {(0, 0): {'starts': [817.1469304232438, 822.64684197257], 'stops': [817.2890435688072, 822.7876269578944]}, (0, 1): {'starts': [1151.1465500156003, 1156.6465933568002], 'stops': [1151.2482070816761, 1156.748250422876]}, (0, 2): {'starts': [1487.849195548745, 1493.349226222291], 'stops': [1487.9294591542682, 1493.4294898278142]}},
    'VertexBarrelCollection': {(0, 0): {'starts': [30.164054331188147], 'stops': [31.339216205416236]}, (0, 1): {'starts': [32.21401583261179], 'stops': [33.3178328434028]}, (0, 2): {'starts': [51.16372563824798], 'stops': [52.91530571876513]}, (0, 4): {'starts': [74.16391955511261], 'stops': [75.47527320758213]}, (0, 6): {'starts': [102.1644051877596], 'stops': [103.12008552250565]}}
}




modules_dir = {

    "InnerTrackerBarrelCollection": buildInnerTrackerBarrelModules(),
    "OuterTrackerBarrelCollection": buildOuterTrackerBarrelModules(),
    "VertexBarrelCollection": buildVertexBarrelModules(),
}





def apply_material_map_hybrid(samples_dir, material_map, col_name, master_feature_indices_dict):

    """
    material_map is in r-phi. We only use it for the endcaps (tbd)
    """

    data = samples_dir[col_name]

    phi = data[:, master_feature_indices_dict[col_name]["phi"]]
    side = data[:, master_feature_indices_dict[col_name]["side"]].astype(int)
    layer_id = data[:, master_feature_indices_dict[col_name]["layer"]].astype(int)

    loc_x = data[:, master_feature_indices_dict[col_name]["r"]]*np.cos(phi)
    loc_y = data[:, master_feature_indices_dict[col_name]["r"]]*np.sin(phi)

   

    if "Barrel" in col_name:
        tangent_coord = data[:, master_feature_indices_dict[col_name]["r"]]
        layer_coord = data[:, master_feature_indices_dict[col_name]["r"]]
        geom = r_side_layer_map_barrel[col_name]

        
        if col_name in ["InnerTrackerBarrelCollection"]:
            mask_tangent_coord_phi =  make_barrel_mask(loc_x, loc_y,modules_dir[col_name], corridor_width=1.55)
        if col_name in ["OuterTrackerBarrelCollection"]:
            mask_tangent_coord_phi =  make_barrel_mask(loc_x, loc_y,modules_dir[col_name], corridor_width=1.9)
        elif col_name in ["VertexBarrelCollection"]:
            mask_tangent_coord_phi =  make_barrel_mask(loc_x, loc_y,modules_dir[col_name], corridor_width=0.44)

        
    elif "Endcap" in col_name: 
        tangent_coord = data[:, master_feature_indices_dict[col_name]["r"]] 
        layer_coord = data[:, master_feature_indices_dict[col_name]["z"]]
        geom = z_side_layer_map_endcaps[col_name]

        if col_name == "InnerTrackerEndcapCollection":
            mask_tangent_coord_phi = make_polygonal_annulus_mask(loc_x, loc_y, None, INNER_ENDCAP_DISKS, z_tolerance=None, nsides=26)
        elif col_name == "OuterTrackerEndcapCollection":
            mask_tangent_coord_phi = make_polygonal_annulus_mask(loc_x, loc_y, None, OUTER_ENDCAP_DISKS, z_tolerance=None, nsides=48, rotation_deg=3.75)
        elif col_name == "VertexEndcapCollection":
            mask_tangent_coord_phi = make_polygonal_annulus_mask(loc_x, loc_y, None, VERTEX_ENDCAP_DISKS, z_tolerance=None, nsides=16)
         


    # Define the geometric map (as a function of layer)
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
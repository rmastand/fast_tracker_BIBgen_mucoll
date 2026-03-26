import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial import cKDTree


def make_geometry_map(
    source_df,
    collections_list,
    save_path,
    num_events_per_col=1_000_000,
    
):

    geometry_map = {}
    for collection_name in collections_list:

        print(f"Analyzing collection {collection_name}...")
    
        df_col = source_df[source_df["collection"] == collection_name]
        
        for i in tqdm(range(num_events_per_col)):
        
            event = df_col.iloc[i]
        
            key = (
                event["collection"],
                event["system"],
                event["side"],
                event["layer"],
                event["module"],
                event["sensor"],
                event["cellid0"],
                event["x"],
                event["y"],
                event["z"],
            )
        
            geometry_map[key] = True
    
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Valid CellID combinations:\n")
        for cell_tuple in sorted(geometry_map):
            f.write(str(cell_tuple) + "\n")
    
    print(f"Saved {len(geometry_map)} unique hits to {save_path}.txt")
    

def build_dist_tree_from_map(
    path_to_geometry_map,
):

    geom_map = defaultdict(list)  # collection -> (x,y,z,cellid0)
    
    i = 0
    with open(path_to_geometry_map, "r", encoding="utf-8") as f:
        for line in f:
    
            if line.startswith("("):
                tup = eval(line)
    
                x = float(tup[7])
                y = float(tup[8])
                z = float(tup[9])
                col_name = tup[0]
                cellid0 = int(tup[6])
                layer = int(tup[3])
                geom_map[col_name].append((x, y, z, cellid0, layer))
    
    print(
        f"Loaded geometry map with {sum(len(v) for v in geom_map.values())} hits across {len(geom_map)} collections"
    )
    
    
    # Build KDTree for each collection
    tree_map = {}
    cellid_layer_map = {}
    #position_map = {}
    
    for col, hits in geom_map.items():
    
    
        coords = np.array([(x, y, z) for x, y, z, _, _ in hits])
        tree_map[col] = cKDTree(coords)
        cellid_layer_map[col] = np.array([(cid, layer) for _, _, _, cid, layer in hits], dtype=np.int64)
        #position_map[col] = coords


    return tree_map, cellid_layer_map #, position_map
    



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
        
        for i in tqdm(range(min(num_events_per_col, len(df_col)))):
        
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


import re

def parse_numeric(s):
    s = s.strip()

    # Remove numpy scalar wrappers:
    # np.float64(123.4) -> 123.4
    # np.int64(3) -> 3
    s = re.sub(r"np\.\w+\((.*?)\)", r"\1", s)

    # Decide int vs float
    try:
        return int(s)
    except ValueError:
        return float(s)

        

def build_dist_tree_from_map(
    path_to_geometry_map,
):

    geom_map = defaultdict(list)  # collection -> (x,y,z,cellid0)
    i = 0
    
    with open(path_to_geometry_map, "r", encoding="utf-8") as f:
        for line in f:
            i += 1
            if i % 100_000 == 0:
                print(i)
            
    
            if line[0] == "(":
                #tup = eval(line)
                tup = line.strip()[1:-1].split(",")
    
                col_name = tup[0].strip().strip("'")

                x = parse_numeric(tup[7])
                y = parse_numeric(tup[8])
                z = parse_numeric(tup[9])
                
                cellid0 = parse_numeric(tup[6])
                layer = parse_numeric(tup[3])
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
    



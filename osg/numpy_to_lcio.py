#!/usr/bin/env python3
from __future__ import annotations

from html import parser

from tqdm import tqdm
from scipy.spatial import cKDTree
from dataclasses import dataclass
import argparse
import math
import os
from array import array
from pathlib import Path
from typing import Any
import dd4hep
import ROOT
import csv
import numpy as np
import pyLCIO
from pyLCIO import IOIMPL, IMPL, EVENT
from collections import defaultdict
import ROOT
from ROOT import Math
import os
#from tqdm import tqdm

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--INPUT_SUFFIX", type=str, default="", help="Path to the samples directory")
parser.add_argument("--OUTPUT_PATH", type=str, default="/scratch/rrm39/v7_reco/slcio/output_flow_18_06.slcio", help="Output LCIO file path")
parser.add_argument("--USE_TREE_CELL_ID", action="store_true", help="If set, read tree_cell_id from column 9 and run mismatch diagnostics against the geometry-assigned cell ID")
args = parser.parse_args()

OUTPUT_PATH = args.OUTPUT_PATH
INPUT_SUFFIX = args.INPUT_SUFFIX
USE_TREE_CELL_ID = args.USE_TREE_CELL_ID

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

_log_path = Path(f"/scratch/rrm39/v7_reco/recoBIB/flow_samples/{INPUT_SUFFIX}/run_log.txt")
_log_path.parent.mkdir(parents=True, exist_ok=True)
_log_file = open(_log_path, "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)

# Load hit arrays
collections = [
     "OuterTrackerBarrelCollection",
     "OuterTrackerEndcapCollection",
     "InnerTrackerBarrelCollection",
     "InnerTrackerEndcapCollection",
     "VertexBarrelCollection",
     "VertexEndcapCollection"    
   
]
NUM_EVENTS = 1
TRACKER_CELL_ID_ENCODING = "system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8"


###########################################################
#
#
# START CELL ID ASSIGNMENT FUNCTIONS
#
#
###########################################################


# Set to None to disable reference lookup.
REF_H5_PATH: Path | None = None

STATUS_NONE = 0
STATUS_UNIQUE = 1
STATUS_MULTI = 2
STATUS_RESCUED = 3

# Numerical tolerance for sensor boundaries and xyz round-trip noise.
FP_EPS_MM = 1e-6

@dataclass
class SensorGeometry:
    cellids: np.ndarray
    systems: np.ndarray
    centers: np.ndarray
    axes: np.ndarray
    half_lengths: np.ndarray
    shape_names: np.ndarray
    trd2_params: np.ndarray
    trees: dict[int, cKDTree]
    tree_indices: dict[int, np.ndarray]

@dataclass
class ReferenceLookup:
    tree: cKDTree
    cellids: np.ndarray


def decode_cellid(cellid: int) -> tuple[int, int, int, int, int]:
    raw_side = (int(cellid) >> 5) & 0x3
    side = raw_side - 4 if raw_side >= 2 else raw_side
    return (
        int(cellid) & 0x1F,
        side,
        (int(cellid) >> 7) & 0x3F,
        (int(cellid) >> 13) & 0x7FF,
        (int(cellid) >> 24) & 0xFF,
    )


def load_sensor_geometry(path: Path | str) -> SensorGeometry:
    data = np.load(path, allow_pickle=False)
    systems = data["systems"].astype(np.int16)
    centers = data["centers_mm"].astype(np.float64)

    trees: dict[int, cKDTree] = {}
    tree_indices: dict[int, np.ndarray] = {}
    for system in np.unique(systems):
        system_id = int(system)
        indices = np.flatnonzero(systems == system_id)
        trees[system_id] = cKDTree(centers[indices])
        tree_indices[system_id] = indices

    return SensorGeometry(
        cellids=data["cellids"].astype(np.int64),
        systems=systems,
        centers=centers,
        axes=data["axes"].astype(np.float64),
        half_lengths=data["half_lengths_mm"].astype(np.float64),
        shape_names=data["shape_names"].astype(str),
        trd2_params=data["trd2_params_mm"].astype(np.float64),
        trees=trees,
        tree_indices=tree_indices,
    )


def context_center_cm(context):
    point = context.localToWorld(array("d", [0.0, 0.0, 0.0]))
    return (float(point.X()), float(point.Y()), float(point.Z()))


def local_to_master(matrix, xyz_cm):
    src = array("d", xyz_cm)
    dst = array("d", [0.0, 0.0, 0.0])
    matrix.LocalToMaster(src, dst)
    return np.array(dst, dtype=np.float64)


def shape_half_lengths_cm(shape):
    if shape is None:
        return np.nan, np.nan, np.nan, False
    if hasattr(shape, "ComputeBBox"):
        shape.ComputeBBox()

    vals = []
    ok = True
    for name in ("GetDX", "GetDY", "GetDZ"):
        if hasattr(shape, name):
            vals.append(float(getattr(shape, name)()))
        else:
            vals.append(np.nan)
            ok = False
    return vals[0], vals[1], vals[2], ok


def trd2_params_mm(shape):
    if shape is None or not hasattr(shape, "GetDx1"):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    return (
        float(shape.GetDx1()) * 10.0,
        float(shape.GetDx2()) * 10.0,
        float(shape.GetDy1()) * 10.0,
        float(shape.GetDy2()) * 10.0,
        float(shape.GetDz()) * 10.0,
    )

MAIA_xml = os.path.join(os.getenv("MUCOLL_GEO"), "k4geo/MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml")

def build_sensor_geometry(path: Path) -> None:
    import dd4hep
    import ROOT

    if "MUCOLL_GEO" not in os.environ:
        raise RuntimeError("MUCOLL_GEO is not set")

    path.parent.mkdir(parents=True, exist_ok=True)
    detector = dd4hep.Detector.getInstance()
    detector.fromXML(MAIA_xml)
    geom = ROOT.gGeoManager
    nav = geom.GetCurrentNavigator()
    vm = detector.volumeManager()

    by_cellid: dict[int, str] = {}
    for system in range(1, 7):
        subdetector = vm.subdetector(system)
        for pair in subdetector.ptr().volumes:
            cellid = int(pair.first)
            if decode_cellid(cellid)[0] != system:
                continue
            if cellid in by_cellid:
                continue
            # Use the sensitive placement center only to recover its TGeo path.
            x, y, z = context_center_cm(pair.second)
            if not nav.FindNode(x, y, z):
                continue
            found_path = geom.GetPath()
            # Keep only paths deep enough to identify an individual sensor.
            if found_path != "/world_volume_1" and found_path.count("/") >= 3:
                by_cellid[cellid] = found_path

    cellids = []
    systems = []
    centers = []
    axes = []
    half_lengths = []
    shape_names = []
    trd2_params = []
    bad_half_lengths = 0

    for cellid, found_path in sorted(by_cellid.items()):
        if not geom.cd(found_path):
            continue

        matrix = geom.GetCurrentMatrix()
        node = geom.GetCurrentNode()
        volume = node.GetVolume() if node else None
        shape = volume.GetShape() if volume else None

        # Store the sensor frame in global coordinates for fast containment tests.
        center_cm = local_to_master(matrix, (0.0, 0.0, 0.0))

        axis = np.vstack(
            [
                local_to_master(matrix, (1.0, 0.0, 0.0)) - center_cm,
                local_to_master(matrix, (0.0, 1.0, 0.0)) - center_cm,
                local_to_master(matrix, (0.0, 0.0, 1.0)) - center_cm,
            ]
        )
        norms = np.linalg.norm(axis, axis=1)
        axis = axis / np.where(norms[:, None] == 0.0, 1.0, norms[:, None])

        system = decode_cellid(cellid)[0]
        dx, dy, dz, half_lengths_ok = shape_half_lengths_cm(shape)
        if not half_lengths_ok:
            bad_half_lengths += 1
        cellids.append(cellid)
        systems.append(system)
        centers.append(center_cm * 10.0)
        axes.append(axis)
        half_lengths.append(np.array([dx, dy, dz], dtype=np.float64) * 10.0)
        shape_names.append(shape.ClassName() if shape else "")
        trd2_params.append(np.array(trd2_params_mm(shape), dtype=np.float64))

    if not cellids:
        raise RuntimeError("No sensitive sensor geometry was found")

    np.savez_compressed(
        path,
        cellids=np.array(cellids, dtype=np.int64),
        systems=np.array(systems, dtype=np.int16),
        centers_mm=np.vstack(centers).astype(np.float64),
        axes=np.stack(axes).astype(np.float64),
        half_lengths_mm=np.vstack(half_lengths).astype(np.float64),
        shape_names=np.array(shape_names, dtype=str),
        trd2_params_mm=np.vstack(trd2_params).astype(np.float64),
    )
    print(f"built sensor geometry: {path} ({len(cellids)} sensors)")
    if bad_half_lengths:
        print(f"warning: {bad_half_lengths} sensors have incomplete shape half lengths")


def load_or_build_sensor_geometry(path: Path) -> SensorGeometry:
    if not path.exists():
        build_sensor_geometry(path)
    return load_sensor_geometry(path)


XYZ_KEY_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])


def xyz_keys(xyz):
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    return xyz.view(XYZ_KEY_DTYPE).reshape(-1)


def load_h5_reference_lookup(path: Path, collection: str) -> ReferenceLookup:
    import pandas as pd

    xyz_chunks = []
    cellid_chunks = []
    where = f'collection == "{collection}"'
    for chunk in pd.read_hdf(path, key="df", where=where, columns=["x", "y", "z", "cellid0"], chunksize=500_000):
        xyz_chunks.append(chunk[["x", "y", "z"]].to_numpy(dtype=np.float64))
        cellid_chunks.append(chunk["cellid0"].to_numpy(dtype=np.int64))
    if not xyz_chunks:
        return build_reference_lookup(np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64))
    return build_reference_lookup(np.concatenate(xyz_chunks), np.concatenate(cellid_chunks))


def build_reference_lookup(xyz: np.ndarray, cellids: np.ndarray) -> ReferenceLookup:
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    cellids = np.asarray(cellids, dtype=np.int64)

    keys = xyz_keys(xyz)
    order = np.argsort(keys, order=("x", "y", "z"))
    keys = keys[order]
    xyz = xyz[order]
    cellids = cellids[order]

    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    stops = np.r_[starts[1:], len(keys)]
    keep = np.ones(len(starts), dtype=bool)
    for i in np.flatnonzero((stops - starts) > 1):
        start = starts[i]
        stop = stops[i]
        if np.any(cellids[start:stop] != cellids[start]):
            keep[i] = False

    kept = starts[keep]
    return ReferenceLookup(tree=cKDTree(xyz[kept]), cellids=cellids[kept])


def lookup_reference_cellids(ref: ReferenceLookup, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Nearest neighbor within FP_EPS_MM -- r/phi round trips aren't bit exact.
    dist, index = ref.tree.query(xyz, k=1, workers=-1)
    found = dist <= FP_EPS_MM
    cellids = np.full(len(xyz), -1, dtype=np.int64)
    cellids[found] = ref.cellids[index[found]]
    return cellids, found


# def pack_cellid(system: int, side: int, layer: int, module: int, sensor: int) -> int:
#     return (
#         (system & 0x1F)
#         | ((side & 0x3) << 5)
#         | ((layer & 0x3F) << 7)
#         | ((module & 0x7FF) << 13)
#         | ((sensor & 0xFF) << 24)
#     )
def decode_cellids(cellids: np.ndarray) -> np.ndarray:
    cellids = cellids.astype(np.int64, copy=False)
    raw_side = (cellids >> 5) & 0x3
    side = np.where(raw_side >= 2, raw_side - 4, raw_side)
    return np.column_stack(
        (
            cellids & 0x1F,
            side,
            (cellids >> 7) & 0x3F,
            (cellids >> 13) & 0x7FF,
            (cellids >> 24) & 0xFF,
        )
    ).astype(np.int64)


def _candidate_indices(xyz, system, geom, k):
    base = geom.tree_indices[system]
    kk = min(k, len(base))
    _dist, local_indices = geom.trees[system].query(xyz, k=kk, workers=-1)
    local_indices = np.atleast_2d(local_indices)
    if local_indices.shape[0] != len(xyz):
        local_indices = local_indices.T
    return base[local_indices]


def _local_coordinates(xyz, candidates, geom):
    diff = xyz[:, None, :] - geom.centers[candidates]
    return np.einsum("nkij,nkj->nki", geom.axes[candidates], diff)


def _effective_half_lengths(local, candidates, geom):
    half = geom.half_lengths[candidates]
    hx = half[:, :, 0].copy()
    hy = half[:, :, 1].copy()
    hz = half[:, :, 2].copy()

    trd2 = geom.shape_names[candidates] == "TGeoTrd2"
    if np.any(trd2):
        rows, cols = np.where(trd2)
        params = geom.trd2_params[candidates[rows, cols]]
        dz = params[:, 4]
        frac = np.clip((local[rows, cols, 2] / np.where(dz == 0.0, 1.0, dz) + 1.0) * 0.5, 0.0, 1.0)
        hx[rows, cols] = params[:, 0] * (1.0 - frac) + params[:, 1] * frac
        hy[rows, cols] = params[:, 2] * (1.0 - frac) + params[:, 3] * frac
        hz[rows, cols] = dz

    return hx, hy, hz


def find_contained_sensors(
    xyz: np.ndarray,
    system: int,
    geom: SensorGeometry,
    k: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = _candidate_indices(xyz, system, geom, k)
    local = _local_coordinates(xyz, candidates, geom)
    hx, hy, hz = _effective_half_lengths(local, candidates, geom)
    inside = (
        (np.abs(local[:, :, 0]) <= hx + FP_EPS_MM)
        & (np.abs(local[:, :, 1]) <= hy + FP_EPS_MM)
        & (np.abs(local[:, :, 2]) <= hz + FP_EPS_MM)
    )
    return candidates, inside


def _classify(
    xyz: np.ndarray,
    system: int,
    geom: SensorGeometry,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates, inside = find_contained_sensors(xyz, system, geom, k)


    n_inside = inside.sum(axis=1)

    status = np.full(len(xyz), STATUS_NONE, dtype=np.int8)
    assigned = np.full(len(xyz), -1, dtype=np.int64)

    unique_rows = np.flatnonzero(n_inside == 1)
    if len(unique_rows):
        first = np.argmax(inside[unique_rows], axis=1)
        assigned[unique_rows] = geom.cellids[candidates[unique_rows, first]]
        status[unique_rows] = STATUS_UNIQUE

    status[n_inside > 1] = STATUS_MULTI

    # ---- MULTI CASE (NEW LOGIC) ----
    multi_rows = np.flatnonzero(n_inside > 1)
    if len(multi_rows):

        cands = candidates[multi_rows]
        local = _local_coordinates(xyz[multi_rows], cands, geom)

        hx, hy, hz = _effective_half_lengths(local, cands, geom)

        # normalized "depth score" (bigger = more safely inside)
        nx = np.abs(local[:, :, 0]) / (hx + 1e-12)
        ny = np.abs(local[:, :, 1]) / (hy + 1e-12)
        nz = np.abs(local[:, :, 2]) / (hz + 1e-12)

        score = np.maximum.reduce([nx, ny, nz])  # L∞ normalized distance

        best = np.argmin(score, axis=1)

        assigned[multi_rows] = geom.cellids[cands[np.arange(len(multi_rows)), best]]
        status[multi_rows] = STATUS_MULTI


    return assigned, status, candidates, inside






# detector = dd4hep.Detector.getInstance()
# xml = os.path.join(os.getenv("MUCOLL_GEO"), "k4geo/MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml")
# detector.fromXML(xml)
# path_to_cellid = load_or_build_path_cellid_map(
#     detector,
#     Path("/scratch/rrm39/tutorial2024/MuC-Tutorial/analysis/BIBAI/geomap_path_cellid.csv"),
#     False,
# )
# nav = ROOT.gGeoManager.GetCurrentNavigator()


###########################################################
#
#
# END CELL ID ASSIGNMENT FUNCTIONS
#
#
###########################################################

# Open LCIO writer
writer = IOIMPL.LCFactory.getInstance().createLCWriter()
writer.open(OUTPUT_PATH, EVENT.LCIO.WRITE_NEW)




# build detector object for cellID assignment
# detector = dd4hep.Detector.getInstance()
# xml = os.path.join(os.getenv("MUCOLL_GEO"), "k4geo/MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml")
# detector.fromXML(xml)
# nav = ROOT.gGeoManager.GetCurrentNavigator()

system_id_dict = {
    "VertexBarrelCollection": 1,   
    "VertexEndcapCollection": 2,
    "InnerTrackerBarrelCollection": 3,
    "InnerTrackerEndcapCollection": 4,
    "OuterTrackerBarrelCollection": 5,
    "OuterTrackerEndcapCollection": 6,
}

path_to_geo_map = Path("/scratch/rrm39/tutorial2024/MuC-Tutorial/analysis/BIBAI/geomap_sensor_geometry.npz")
geom = load_or_build_sensor_geometry(path_to_geo_map)
print(f"loaded sensor geometry: {path_to_geo_map}")
k = 64

for evt_num in range(NUM_EVENTS):

    evt = IMPL.LCEventImpl()
    evt.setEventNumber(evt_num)
    evt.setRunNumber(1)

    for COLLECTION_NAME in collections:

        print(f"Processing collection: {COLLECTION_NAME}")

        num_hits_total = 0
        num_mismatched_cell_ids = 0


        hits_array = np.load(f"/scratch/rrm39/v7_reco/recoBIB/flow_samples/{INPUT_SUFFIX}/{COLLECTION_NAME}.npy")
  
 
        col = IMPL.LCCollectionVec(EVENT.LCIO.SIMTRACKERHIT)

        col.getParameters().setValue(
            "CellIDEncoding",
            TRACKER_CELL_ID_ENCODING
        )



         # loge, t, r, phi, z, side, layer, module, sensor, cell_id 

        NN = 5000
        all_tree_cell_ids_valid, all_tree_cell_ids_invalid = [], []
        all_geo_cell_ids_valid, all_geo_cell_ids_invalid = [], []
        all_valid_masks = []

        # iterate through the entire hits array with chunk size NN
        for i in tqdm(range(0, len(hits_array), NN)):
            chunk = hits_array[i:i+NN]
            NN_chunk = len(chunk)
            
            # first assign cell ID
            loge = chunk[:, 0].astype(np.float64)
            time = chunk[:, 1].astype(np.float64)
            radius = chunk[:, 2].astype(np.float64)
            phi = chunk[:, 3].astype(np.float64)
            z = chunk[:, 4].astype(np.float64)
            edep = np.exp(loge)
            xyz = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
            systems = np.full(len(chunk), system_id_dict[COLLECTION_NAME], dtype=np.int16)

            tree_cell_id = chunk[:, 9].astype(np.int64) if USE_TREE_CELL_ID else None


            assigned = np.full(len(chunk), -1, dtype=np.int64)
            status = np.full(len(chunk), STATUS_NONE, dtype=np.int8)

            for system in np.unique(systems):
                part = np.flatnonzero(systems == system)
                assigned_part, status_part, _candidates, _inside = _classify(xyz[part], int(system), geom, k)
                assigned[part] = assigned_part
                status[part] = status_part

            valid = (status == STATUS_UNIQUE) | (status == STATUS_RESCUED) | (status == STATUS_MULTI)
            all_geo_cell_ids_valid.append(assigned[valid])
            all_geo_cell_ids_invalid.append(assigned[~valid])
            all_valid_masks.append(valid)
            if USE_TREE_CELL_ID:
                all_tree_cell_ids_valid.append(tree_cell_id[valid])
                all_tree_cell_ids_invalid.append(tree_cell_id[~valid])

            valid_xyz = xyz[valid]
            valid_edep = edep[valid]
            valid_time = time[valid]
            valid_assigned = assigned[valid]

            for hit_i in range(len(valid_edep)):
                num_hits_total += 1

                #logE, t, r, phi, z, side, layer, module, sensor, _ = chunk[valid][hit_i]
                # assigned_cellid = pack_cell_id(system_id_dict[COLLECTION_NAME], int(side), int(layer), int(module), int(sensor))

                #x = r * np.cos(phi)
                #y = r * np.sin(phi)

                # # Generated positions are in mm; TGeo navigator expects cm.
                # node = nav.FindNode(x / 10.0, y / 10.0, z / 10.0)
                
                # actual_cellid = path_to_cellid.get(ROOT.gGeoManager.GetPath()) if node else None

                
                # if assigned_cellid != actual_cellid:
                #     print(f"Warning: Assigned cell ID {assigned_cellid} does not match actual cell ID {actual_cellid} for hit at (x={x:.2f}, y={y:.2f}, z={z:.2f})")
                #     num_mismatched_cell_ids += 1

                simhit = IMPL.SimTrackerHitImpl()

                simhit.setEDep(valid_edep[hit_i])
                simhit.setTime(valid_time[hit_i])
                simhit.setPosition(np.array([valid_xyz[hit_i][0], valid_xyz[hit_i][1], valid_xyz[hit_i][2]], dtype=np.float64))
                simhit.setCellID0(int(valid_assigned[hit_i]))

                col.addElement(simhit)

        all_geo_cell_ids_valid = np.concatenate(all_geo_cell_ids_valid)
        all_geo_cell_ids_invalid = np.concatenate(all_geo_cell_ids_invalid)
        all_valid_masks = np.concatenate(all_valid_masks)

        # print out the number of invalid hits and the percentage of invalid hits
        print(f"Total valid hits = {len(all_geo_cell_ids_valid)}, Total invalid hits = {len(all_geo_cell_ids_invalid)}, percent invalid = {100.0 * len(all_geo_cell_ids_invalid) / (len(all_geo_cell_ids_valid) + len(all_geo_cell_ids_invalid))}")

        if USE_TREE_CELL_ID:
            all_tree_cell_ids_valid = np.concatenate(all_tree_cell_ids_valid)
            all_tree_cell_ids_invalid = np.concatenate(all_tree_cell_ids_invalid)

            mismatch_mask = all_geo_cell_ids_valid != all_tree_cell_ids_valid
            num_mismatched_cell_ids_valid = np.sum(mismatch_mask)
            num_hits_valid_total = len(all_geo_cell_ids_valid)
            print(f"Total valid hits = {num_hits_valid_total}, Mismatched cell IDs = {num_mismatched_cell_ids_valid}, Mismatch percent = {100.0 * num_mismatched_cell_ids_valid / num_hits_valid_total}")

            mismatch_indices = np.flatnonzero(mismatch_mask)

            labels_truth = decode_cellids(all_tree_cell_ids_valid)
            labels_geo = decode_cellids(all_geo_cell_ids_valid)

            radius = hits_array[:, 2].astype(np.float64)
            phi = hits_array[:, 3].astype(np.float64)
            z = hits_array[:, 4].astype(np.float64)
            tmp_xyz = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))

            xyz_mismatch = tmp_xyz[all_valid_masks]

            for idx in mismatch_indices[::10]:
                print(
                    f"Hit (index {idx}): "
                    f"Position = {xyz_mismatch[idx]}"
                )

                xyz = xyz_mismatch[idx][None, :]

                candidates = _candidate_indices(xyz, system, geom, k=k)
                local = _local_coordinates(xyz, candidates, geom)

                pred_cellid = all_geo_cell_ids_valid[idx]
                pred_idx = np.where(geom.cellids == pred_cellid)[0]

                if len(pred_idx):
                    j = pred_idx[0]
                    diff = xyz[0] - geom.centers[j]
                    local_pred = geom.axes[j] @ diff

                    print("Predicted sensor:")
                    print("  cellid =", pred_cellid)
                    print("  decoded =", decode_cellid(pred_cellid))
                    print("  local =", local_pred)
                    print("  center =", geom.centers[j])

                truth_cellid = all_tree_cell_ids_valid[idx]
                truth_idx = np.where(geom.cellids == truth_cellid)[0]

                if len(truth_idx):
                    j = truth_idx[0]
                    diff = xyz[0] - geom.centers[j]
                    local_truth = geom.axes[j] @ diff

                    print("Truth sensor:")
                    print("  cellid =", truth_cellid)
                    print("  decoded =", decode_cellid(truth_cellid))
                    print("  local =", local_truth)
                    print("  center =", geom.centers[j])

                print()
                print()

            print(f"Total invalid hits = {len(all_geo_cell_ids_invalid)}, percent invalid = {100.0 * len(all_geo_cell_ids_invalid) / (len(all_geo_cell_ids_valid) + len(all_geo_cell_ids_invalid))}")
            for invalid_hit in range(len(all_geo_cell_ids_invalid[:20])):
                labels_truth = decode_cellids(all_tree_cell_ids_invalid).astype(np.float64)
                invalid_hit_xyz = tmp_xyz[~all_valid_masks][invalid_hit]
                truth_cellid = all_tree_cell_ids_invalid[invalid_hit]
                print(f"Hit {invalid_hit}: Truth cell ID = {truth_cellid}, Truth decoded = {labels_truth[invalid_hit]}. Position = {invalid_hit_xyz}")

                truth_index = np.where(geom.cellids == truth_cellid)[0]
                truth_index = truth_index[0]

                candidate_indices = _candidate_indices(
                    invalid_hit_xyz[None, :],
                    system_id_dict[COLLECTION_NAME],
                    geom,
                    k=64,
                )

                local = _local_coordinates(
                    invalid_hit_xyz[None, :],
                    candidate_indices,
                    geom,
                )

                hx, hy, hz = _effective_half_lengths(local, candidate_indices, geom)

                inside = (
                    (np.abs(local[:,:,0]) <= hx + FP_EPS_MM)
                    &
                    (np.abs(local[:,:,1]) <= hy + FP_EPS_MM)
                    &
                    (np.abs(local[:,:,2]) <= hz + FP_EPS_MM)
                )

                print("Inside cellids:")
                print(geom.cellids[candidate_indices[0][inside[0]]])
                inside_indices = candidate_indices[0][inside[0]]

                for idx in inside_indices:
                    print(
                        "cellid =", geom.cellids[idx],
                        "center =", geom.centers[idx],
                        "shape =", geom.shape_names[idx],
                    )

                    diff = invalid_hit_xyz - geom.centers[idx]
                    local = geom.axes[idx] @ diff

                    print("local =", local)

                candidate_indices = _candidate_indices(
                    invalid_hit_xyz[None, :],
                    system_id_dict[COLLECTION_NAME],
                    geom,
                    k=64,
                )[0]

                candidate_cellids = geom.cellids[candidate_indices]

                print("Truth in candidates?", truth_cellid in candidate_cellids)

                if truth_cellid in candidate_cellids:
                    print("    Candidate rank:",
                        np.where(candidate_cellids == truth_cellid)[0][0])
                center = geom.centers[truth_index]
                axes = geom.axes[truth_index]
                diff = invalid_hit_xyz - center
                local = axes @ diff
                half = geom.half_lengths[truth_index]

                hx = half[0]
                hy = half[1]
                hz = half[2]

                if geom.shape_names[truth_index] == "TGeoTrd2":
                    params = geom.trd2_params[truth_index]

                    dx1, dx2, dy1, dy2, dz = params

                    frac = np.clip((local[2] / dz + 1.0) * 0.5, 0.0, 1.0)

                    hx = dx1 * (1.0 - frac) + dx2 * frac
                    hy = dy1 * (1.0 - frac) + dy2 * frac
                    hz = dz

                inside = (
                    abs(local[0]) <= hx + FP_EPS_MM and
                    abs(local[1]) <= hy + FP_EPS_MM and
                    abs(local[2]) <= hz + FP_EPS_MM
                )

                print(f"    local = {local}")
                print(f"    center = {center}")
                #print(f"    half lengths = ({hx}, {hy}, {hz})")
                print(f"    inside truth sensor? {inside}")

                print()




        

        evt.addCollection(col, COLLECTION_NAME)
        #print(f"Total hits = {num_hits_total}, Mismatched cell IDs = {num_mismatched_cell_ids}, Mismatch percent = {100.0 * num_mismatched_cell_ids / num_hits_total}")

    writer.writeEvent(evt)

writer.close()

print(f"Wrote {OUTPUT_PATH} with {NUM_EVENTS} events using closest geometry CellID0")

sys.stdout = sys.__stdout__
_log_file.close()
print(f"Log saved to {_log_path}")

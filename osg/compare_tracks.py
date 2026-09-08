#!/usr/bin/env python3
"""
Compare track properties from any number of sets of SLCIO files.
Edit TRACK_SETS below: each entry is (label, [file, ...]).
Add or remove entries freely — the rest of the script adapts automatically.
"""
import os
import glob
import resource
import gc  # CHANGE: added so we can explicitly free LCIO reader/event memory after each file.
import sys  # CHANGE: added so the same script can run in parent mode or one-file worker mode.
import subprocess  # CHANGE: added so the parent can launch one fresh Python process per SLCIO file.
import tempfile  # CHANGE: added so worker .npz histogram files can be written to a temporary output directory.
from collections import defaultdict
from math import atan, log, tan, pi

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml
import pyLCIO
from pyLCIO import IOIMPL, EVENT, UTIL
DPI = 300
plt.style.use("science.mplstyle")

_osg_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs_osg.yaml")
with open(_osg_cfg_path) as _f:
    _osg_cfg = yaml.safe_load(_f)

# ── User configuration ─────────────────────────────────────────────────────

TRACK_COLLECTIONS = ["SelectedTracks", "SiTracksDeduped"] #  ["SelectedTracks", "SiTracksDeduped"]
OUTPUT_DIR        = _osg_cfg["COMPARE_OUTPUT_DIR"]
BFIELD            = 5.0   # Tesla
FILE_SUFFIX       = "diff"    # appended before .pdf, e.g. "_v2"
RATIO_REF         = "Full Simulation"  # denominator label for the ratio panel

# Each entry: (label, [list of .slcio files])
TRACK_SETS = [

   


    ("Full Simulation", [
        [
            f"{_osg_cfg['SLCIO_DIR']}/output_sim_v7_nugun_0_50_reco{label}_selected_0_30.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_sim_v7_nugun_0_50_reco{label}_selected_30_70.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_sim_v7_nugun_0_50_reco{label}_selected_70_110.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_sim_v7_nugun_0_50_reco{label}_selected_110_150.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_sim_v7_nugun_0_50_reco{label}_selected_150_180.slcio",
        ]
        for label in ["", "_1", "_10", "_11", "_12", "_13", "_14", "_15", "_16", "_17"]
    ]),




    ("GenBIB-Diff", [
        [
            f"{_osg_cfg['SLCIO_DIR']}/output_diff_local_reco{label}{'_reco' if label else ''}_selected_0_30.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_diff_local_reco{label}{'_reco' if label else ''}_selected_30_70.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_diff_local_reco{label}{'_reco' if label else ''}_selected_70_110.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_diff_local_reco{label}{'_reco' if label else ''}_selected_110_150.slcio",
            f"{_osg_cfg['SLCIO_DIR']}/output_diff_local_reco{label}{'_reco' if label else ''}_selected_150_180.slcio",
        ]
                for label in ["", "_1", "_10", "_11", "_12", "_13", "_14", "_15", "_16", "_17"]
    ]),

    ("GenBIB-Flow", [
                [
                    f"{_osg_cfg['SLCIO_DIR']}/output_flow_NCSF_bins8_local_reco{label}{'_reco' if label else ''}_selected_0_30.slcio",
                    f"{_osg_cfg['SLCIO_DIR']}/output_flow_NCSF_bins8_local_reco{label}{'_reco' if label else ''}_selected_30_70.slcio",
                    f"{_osg_cfg['SLCIO_DIR']}/output_flow_NCSF_bins8_local_reco{label}{'_reco' if label else ''}_selected_70_110.slcio",
                    f"{_osg_cfg['SLCIO_DIR']}/output_flow_NCSF_bins8_local_reco{label}{'_reco' if label else ''}_selected_110_150.slcio",
                    f"{_osg_cfg['SLCIO_DIR']}/output_flow_NCSF_bins8_local_reco{label}{'_reco' if label else ''}_selected_150_180.slcio",
                ]
                   for label in ["", "_1", "_10", "_11", "_12", "_13", "_14", "_15", "_16", "_17"]
            ]),

]

# ── Constants ──────────────────────────────────────────────────────────────

HIT_COLLECTIONS = [
    "IBTrackerHits", "IETrackerHits",
    "OBTrackerHits", "OETrackerHits",
    "VBTrackerHits", "VETrackerHits",
    "VXDBarrelHits", "VXDEndcapHits",
    "ITBarrelHits",  "ITEndcapHits",
    "OTBarrelHits",  "OTEndcapHits",
]

SYS_TO_KEY = {
    1: "vxd_barrel_nhits", 2: "vxd_endcap_nhits",
    3: "it_barrel_nhits",  4: "it_endcap_nhits",
    5: "ot_barrel_nhits",  6: "ot_endcap_nhits",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# (key, bins, yunit, xlabel, title, xscale, yscale)
# Output filename is generated at plot time as  {key}__{safe_collection}.pdf
PLOT_SPECS = [
    # ── subdetector hit counts ──────────────────────────────────────────────
    ("vxd_barrel_nhits", np.arange(0, 9) - 0.5,     "hit",  "Hits per Track",        "Vertex Barrel Hits per Track",        "linear", "linear"),
    ("vxd_endcap_nhits", np.arange(0, 9) - 0.5,     "hit",  "Hits per Track",        "Vertex Endcap Hits per Track",        "linear", "linear"),
    ("it_barrel_nhits",  np.arange(0, 6) - 0.5,     "hit",  "Hits per Track",        "Inner Tracker Barrel Hits per Track", "linear", "linear"),
    ("it_endcap_nhits",  np.arange(0, 9) - 0.5,     "hit",  "Hits per Track",        "Inner Tracker Endcap Hits per Track", "linear", "linear"),
    ("ot_barrel_nhits",  np.arange(0, 5) - 0.5,     "hit",  "Hits per Track",        "Outer Tracker Barrel Hits per Track", "linear", "linear"),
    ("ot_endcap_nhits",  np.arange(0, 6) - 0.5,     "hit",  "Hits per Track",        "Outer Tracker Endcap Hits per Track", "linear", "linear"),
    # ── totals & track parameters ───────────────────────────────────────────
    ("total_nhits",      np.arange(0, 25) - 0.5,    "hit",  "Total hits",              "Total Hits per Track",  "linear", "linear"),
    ("eta",              np.linspace(-3, 3, 61),      "",     "Track $\\eta$",           "Track $\\eta$",         "linear", "linear"),
    ("phi",              np.linspace(-pi, pi, 65),   "rad",  "Track $\\phi$ [rad]",     "Track $\\phi$",         "linear", "linear"),
    ("chi2_reduced",     np.linspace(0, 5, 6),        "",     "$\\chi^2/Ndf$",           "Track $\\chi^2/Ndf$",   "linear", "linear"),
    ("d0",               np.linspace(-5, 5, 51),     "mm",   "$d_0$ [mm]",              "Track $d_0$",           "linear", "linear"),
    ("z0",               np.linspace(-25, 25, 101),  "mm",   "$z_0$ [mm]",              "Track $z_0$",           "linear", "linear"),
    ("pt",               np.logspace(-1, 4, 61),     "GeV",  "Track $p_T$ [GeV]",       "Track $p_T$",           "log",    "log"),
]

BINS_BY_KEY = {key: bins for key, bins, _, _, _, _, _ in PLOT_SPECS}

# ── Helpers ────────────────────────────────────────────────────────────────

def eta_from_tanl(tanl):
    theta = pi / 2.0 - atan(tanl)
    return -log(tan(max(theta / 2.0, 1e-10)))


def get_encoding(event):
    """Return CellID encoding string from the first available hit collection."""
    for cname in HIT_COLLECTIONS:
        try:
            col = event.getCollection(cname)
            return col.getParameters().getStringVal(pyLCIO.EVENT.LCIO.CellIDEncoding)
        except Exception:
            continue
    return None


# CHANGE: added helper to fill one histogram bin without storing the original value.
def fill_hist(hist_counts, key, value):
    """Increment the streaming histogram for one value."""
    if not np.isfinite(value):
        return

    bins = BINS_BY_KEY[key]
    ibin = np.searchsorted(bins, value, side="right") - 1

    # CHANGE: match np.histogram/matplotlib behavior by including the right edge of the last bin.
    if ibin == len(bins) - 1 and value == bins[-1]:
        ibin = len(bins) - 2

    # CHANGE: values outside the requested plotting range are ignored, just like hist(..., bins=bins).
    if 0 <= ibin < len(hist_counts[key]):
        hist_counts[key][ibin] += 1.0


# CHANGE: added helper for an approximate median from the already-filled pT histogram.
def hist_median(counts, bins):
    """Approximate the median using the streaming histogram bins."""
    total = np.sum(counts)
    if total <= 0:
        return np.nan

    cdf = np.cumsum(counts)
    ibin = np.searchsorted(cdf, 0.5 * total, side="left")
    ibin = min(max(ibin, 0), len(counts) - 1)

    # CHANGE: use geometric bin center for log-spaced pT bins; arithmetic center otherwise.
    if np.all(bins > 0):
        return np.sqrt(bins[ibin] * bins[ibin + 1])
    return 0.5 * (bins[ibin] + bins[ibin + 1])


def collect_tracks(file_list, label, track_collection):
    """Loop over all files and events; return (track_histograms, layer_hits).

    layer_hits: {sys_id: {layer: total_hit_count_across_all_tracks}}
    Divide by n_tracks to get average hits per track per layer.
    """
    data = {
        key: np.zeros(len(bins) - 1, dtype=np.float64)
        for key, bins, _, _, _, _, _ in PLOT_SPECS
    }

    # CHANGE: keep the total number of accepted tracks explicitly because arrays are now histograms.
    data["n_tracks"] = 0

    # CHANGE: keep finite pT summary stats without storing every pT value.
    data["pt_sum"] = 0.0
    data["pt_count"] = 0

    layer_hits = {sys_id: defaultdict(int) for sys_id in SYS_TO_KEY}

    for fname in file_list:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[{label}] reading {fname}  (RSS so far: {rss_mb:.0f} MB)")
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)

        # CHANGE: use try/finally so the LCIO reader is closed even if a file/event has a problem.
        try:
            for ievt, event in enumerate(reader):
                if ievt % 500 == 0:
                    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    print(f"  [{label}] event {ievt}, tracks so far: {data['n_tracks']}, RSS: {rss_mb:.0f} MB")
                encoding = get_encoding(event)
                if encoding is None:
                    print(f"  event {ievt}: no hit collection found, skipping")
                    continue
                decoder = UTIL.BitField64(encoding)

                try:
                    track_col = event.getCollection(track_collection)
                except Exception:
                    print(f"  event {ievt}: '{track_collection}' not found, skipping")
                    continue

                for track in track_col:
                    ndf = track.getNdf()
                    if ndf <= 0:
                        continue

                    counters = {k: 0 for k in SYS_TO_KEY.values()}
                    for hit in track.getTrackerHits():
                        decoder.setValue(int(hit.getCellID0()))
                        sys = decoder["system"].value()
                        if sys in SYS_TO_KEY:
                            counters[SYS_TO_KEY[sys]] += 1
                            layer_hits[sys][decoder["layer"].value()] += 1

                    # CHANGE: fill histograms immediately instead of appending values to lists.
                    for k, v in counters.items():
                        fill_hist(data, k, v)

                    total_nhits = sum(counters.values())
                    eta = eta_from_tanl(track.getTanLambda())
                    phi = track.getPhi()
                    chi2_reduced = track.getChi2() / float(ndf)
                    d0 = track.getD0()
                    z0 = track.getZ0()
                    omega = track.getOmega()
                    pt = 0.3 * BFIELD / (abs(omega) * 1e3) if omega != 0.0 else np.nan

                    fill_hist(data, "total_nhits", total_nhits)
                    fill_hist(data, "eta", eta)
                    fill_hist(data, "phi", phi)
                    fill_hist(data, "chi2_reduced", chi2_reduced)
                    fill_hist(data, "d0", d0)
                    fill_hist(data, "z0", z0)
                    fill_hist(data, "pt", pt)

                    # CHANGE: count accepted tracks directly since total_nhits is no longer a track array.
                    data["n_tracks"] += 1

                    # CHANGE: update pT mean stats without storing the full pT array.
                    if np.isfinite(pt):
                        data["pt_sum"] += pt
                        data["pt_count"] += 1
        finally:
            # CHANGE: close and delete the reader after every file to avoid retaining LCIO buffers.
            try:
                reader.close()
            except Exception:
                pass
            del reader
            gc.collect()

    return data, layer_hits


# ── Subprocess helpers ─────────────────────────────────────────────────────

def save_worker_output(path, data, layer_hits):
    save_dict = {}

    for key, value in data.items():
        save_dict[f"data__{key}"] = np.asarray(value)

    for sys_id, layer_dict in layer_hits.items():
        layers = np.array(list(layer_dict.keys()), dtype=np.int64)
        counts = np.array(list(layer_dict.values()), dtype=np.float64)
        save_dict[f"layer__{sys_id}__layers"] = layers
        save_dict[f"layer__{sys_id}__counts"] = counts

    np.savez(path, **save_dict)


def load_worker_output(path):
    z = np.load(path, allow_pickle=False)

    data = {}
    for key in z.files:
        if key.startswith("data__"):
            clean_key = key.replace("data__", "", 1)
            value = z[key]
            data[clean_key] = value.item() if value.shape == () else value

    layer_hits = {sys_id: defaultdict(int) for sys_id in SYS_TO_KEY}
    for sys_id in SYS_TO_KEY:
        layer_key = f"layer__{sys_id}__layers"
        count_key = f"layer__{sys_id}__counts"
        if layer_key in z.files and count_key in z.files:
            for layer, count in zip(z[layer_key], z[count_key]):
                layer_hits[sys_id][int(layer)] += float(count)

    return data, layer_hits


def merge_worker_outputs(partial_files):
    merged_data = None
    merged_layer_hits = {sys_id: defaultdict(int) for sys_id in SYS_TO_KEY}

    for path in partial_files:
        data, layer_hits = load_worker_output(path)

        if merged_data is None:
            merged_data = {}
            for key, value in data.items():
                merged_data[key] = value.copy() if isinstance(value, np.ndarray) else value
        else:
            for key, value in data.items():
                merged_data[key] += value

        for sys_id in SYS_TO_KEY:
            for layer, count in layer_hits[sys_id].items():
                merged_layer_hits[sys_id][int(layer)] += float(count)

    return merged_data, merged_layer_hits


def run_worker_if_requested():
    if "--worker" not in sys.argv:
        return False

    label      = sys.argv[sys.argv.index("--label") + 1]
    fname      = sys.argv[sys.argv.index("--file") + 1]
    out_npz    = sys.argv[sys.argv.index("--out") + 1]
    collection = sys.argv[sys.argv.index("--collection") + 1]

    data, layer_hits = collect_tracks([fname], label, collection)
    save_worker_output(out_npz, data, layer_hits)
    print(f"[worker] saved {out_npz}")
    return True


# CHANGE: run worker mode before the parent data-collection block.
if run_worker_if_requested():
    sys.exit(0)


# ── Data collection ────────────────────────────────────────────────────────
# Default: process any (label, collection, event_idx) that lacks a .npz file.
# --plot-only: skip all LCIO processing; use whatever .npz files exist on disk.

def safe(s):
    """Filesystem-safe version of a label or collection name."""
    return s.replace(" ", "_").replace("/", "_").replace("$", "").replace("\\", "").replace(",", "")


NPZ_DIR = os.path.join(OUTPUT_DIR, "histograms")
os.makedirs(NPZ_DIR, exist_ok=True)


def event_npz_path(label, collection, event_idx):
    """Persistent per-(label, collection, event) merged histogram file."""
    return os.path.join(
        NPZ_DIR,
        f"{safe(label)}__{safe(collection)}__event{event_idx:04d}_merged.npz",
    )


if "--plot-only" not in sys.argv:
    tmp_dir = tempfile.mkdtemp(prefix="track_hists_", dir=OUTPUT_DIR)
    print(f"[parent] worker histogram files will be written under {tmp_dir}")

    for collection in TRACK_COLLECTIONS:
        for label, events in TRACK_SETS:
            for event_idx, file_list in enumerate(events):
                out_path = event_npz_path(label, collection, event_idx)
                if os.path.exists(out_path):
                    print(f"[skip] already processed: {out_path}")
                    continue

                print(f"\n[parent] {label} | {collection} | event {event_idx}")
                partial_files = []
                for ifile, fname in enumerate(file_list):
                    tmp_npz = os.path.join(
                        tmp_dir,
                        f"{safe(label)}__{safe(collection)}__ev{event_idx:04d}__f{ifile}.npz",
                    )
                    print(f"  [parent] launching worker: {fname}")
                    subprocess.run(
                        [
                            sys.executable,
                            os.path.abspath(__file__),
                            "--worker",
                            "--label",      label,
                            "--collection", collection,
                            "--file",       fname,
                            "--out",        tmp_npz,
                        ],
                        check=True,
                    )
                    partial_files.append(tmp_npz)

                data, layer_hits = merge_worker_outputs(partial_files)
                save_worker_output(out_path, data, layer_hits)
                print(f"[parent] saved {out_path}  ({data['n_tracks']} tracks)")

print()

# ── Per-event track count text files ──────────────────────────────────────
for collection in TRACK_COLLECTIONS:
    safe_col = safe(collection)
    for label, events in TRACK_SETS:
        for event_idx, _ in enumerate(events):
            npz_path = event_npz_path(label, collection, event_idx)
            if not os.path.exists(npz_path):
                continue
            evt_data, _ = load_worker_output(npz_path)
            n_tracks = int(evt_data["n_tracks"])
            txt_path = os.path.join(
                NPZ_DIR,
                f"{safe(label)}__{safe_col}__event{event_idx:04d}_ntracks.txt",
            )
            with open(txt_path, "w") as f:
                f.write(f"{n_tracks}\n")
            print(f"[ntracks] {label} | {collection} | event {event_idx}: {n_tracks} tracks -> {txt_path}")

print()

# ── Plotting helpers ───────────────────────────────────────────────────────

def overlay_hist(ax, datasets, bins, xlabel, title, xscale="linear", yscale="linear", yunit="", normalize=True, ax_ratio=None, ref_label=None):
    """datasets: list of (hist_counts_array, label, n_tracks) pairs."""
    ref_y = None
    if ax_ratio is not None and ref_label is not None:
        for counts, label, n_tracks in datasets:
            if label == ref_label and n_tracks > 0 and np.sum(counts) > 0:
                ref_y = counts / float(n_tracks) if normalize else counts
                break

    for counts, label, n_tracks in datasets:
        if n_tracks > 0 and np.sum(counts) > 0:
            y = counts / float(n_tracks) if normalize else counts
            ax.stairs(y, bins, label=label, linewidth=1.5)
            if ax_ratio is not None and ref_y is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(ref_y > 0, y / ref_y, np.nan)
                ax_ratio.stairs(ratio, bins, linewidth=1.5)

    ax.set_ylabel("Density" if normalize else "Hits")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend(fontsize=16, loc="best")

    if ax_ratio is not None:
        ax_ratio.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax_ratio.set_xlabel(xlabel)
        ax_ratio.set_xscale(xscale)
        ax_ratio.set_ylabel("Ratio to\nFull Simulation", fontsize=12)
    else:
        ax.set_xlabel(xlabel)


# (display_name, file_key) — output filename: avg_hits_per_layer_{file_key}__{safe_col}.pdf
SYS_NAMES = {
    1: ("VXD Barrel", "vxd_barrel"),
    2: ("VXD Endcap", "vxd_endcap"),
    3: ("IT Barrel",  "it_barrel"),
    4: ("IT Endcap",  "it_endcap"),
    5: ("OT Barrel",  "ot_barrel"),
    6: ("OT Endcap",  "ot_endcap"),
}

# ── Plots — one set of PDFs per TRACK_COLLECTION ───────────────────────────

print()
for collection in TRACK_COLLECTIONS:
    safe_col = safe(collection)

    # Gather all processed events for each label and merge in memory.
    all_data = []
    for label, events in TRACK_SETS:
        pattern = os.path.join(NPZ_DIR, f"{safe(label)}__{safe_col}__event*_merged.npz")
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"[plot] no .npz files for '{label}' | {collection} — skipping")
            continue
        data, layer_hits = merge_worker_outputs(found)
        print(f"[plot] '{label}' | {collection}: {len(found)} event(s), {data['n_tracks']} tracks")
        plot_label = label if label == RATIO_REF else f"{label} (local $\\phi$)"
        all_data.append((plot_label, data, layer_hits))

    if not all_data:
        print(f"[plot] nothing to plot for {collection}\n")
        continue

    # ── Individual PDFs ────────────────────────────────────────────────────

    for key, bins, yunit, xlabel, title, xscale, yscale in PLOT_SPECS:
        fig, (ax_top, ax_ratio) = plt.subplots(
            2, 1, figsize=(6, 5),
            gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        fig.subplots_adjust(hspace=0.05)
        datasets = [(d[key], lbl, d["n_tracks"]) for lbl, d, _ in all_data]
        is_nhits = key.endswith("nhits")
        overlay_hist(ax_top, datasets, bins, xlabel, title, xscale, yscale, yunit=yunit, normalize=True,
                     ax_ratio=ax_ratio, ref_label=RATIO_REF)
        if is_nhits:
            tick_min = int(bins[0] + 0.5)
            tick_max = int(bins[-1] - 0.5)
            ax_top.set_xticks(np.arange(tick_min, tick_max + 1))
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"{key}__{safe_col}{FILE_SUFFIX}.pdf")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"Saved {path}")

    # ── Average hits per track vs. layer — one PDF per subdetector ────────

    for sys_id, (sys_name, sys_key) in SYS_NAMES.items():
        fig, (ax_top, ax_ratio) = plt.subplots(
            2, 1, figsize=(6, 5),
            gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        fig.subplots_adjust(hspace=0.05)
        all_layers = sorted(set().union(*[lh[sys_id].keys() for _, _, lh in all_data]))
        x = np.array(all_layers)
        edges = np.concatenate([x - 0.5, [x[-1] + 0.5]]) if len(x) else np.array([])

        ref_avg = None
        for lbl, d, layer_hits in all_data:
            if lbl == RATIO_REF and d["n_tracks"] > 0:
                lh = layer_hits[sys_id]
                ref_avg = np.array([lh.get(l, 0) / d["n_tracks"] for l in all_layers])
                break

        for lbl, d, layer_hits in all_data:
            n_tracks = d["n_tracks"]
            lh       = layer_hits[sys_id]
            avg = (np.array([lh.get(l, 0) / n_tracks for l in all_layers])
                   if n_tracks else np.zeros(len(x)))
            ax_top.stairs(avg, edges, label=lbl, linewidth=1.5)
            if ref_avg is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(ref_avg > 0, avg / ref_avg, np.nan)
                ax_ratio.stairs(ratio, edges, linewidth=1.5)

        ax_top.set_xticks(x)
        ax_top.set_ylabel("Avg. hits / track")
        ax_top.legend(fontsize=16, loc="best")
        ax_ratio.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax_ratio.set_xticks(x)
        ax_ratio.set_xlabel("Layer")
        ax_ratio.set_ylabel("Ratio to\nFull Simulation", fontsize=12)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"avg_hits_per_layer_{sys_key}__{safe_col}{FILE_SUFFIX}.pdf")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"Saved {path}")

    print()

print("Done.")

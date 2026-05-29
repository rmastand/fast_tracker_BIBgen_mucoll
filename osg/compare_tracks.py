#!/usr/bin/env python3
"""
Compare track properties from any number of sets of SLCIO files.
Edit TRACK_SETS below: each entry is (label, [file, ...]).
Add or remove entries freely — the rest of the script adapts automatically.
"""
import os
import resource
from collections import defaultdict
from math import atan, log, tan, pi

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyLCIO
from pyLCIO import IOIMPL, EVENT, UTIL

# ── User configuration ─────────────────────────────────────────────────────

TRACK_COLLECTION = "SiTracks"
OUTPUT_DIR       = "/scratch/larsonma/MLBibComparison"
BFIELD           = 5.0   # Tesla

# Each entry: (label, [list of .slcio files])
TRACK_SETS = [
    ("ML BIB", [
        "/ospool/uc-shared/project/futurecolliders/brosser/lcio/ml_bib_reco_output_tenpercent.slcio",
    ]),
    ("ML BIB Samples from Shiyu", [
        "/scratch/rrm39/shared/output_reco_shiyu.slcio",
    ]),
    ("Sim BIB", [
        "/ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MAIA_v0/v8/special/nuGun_filtered_0_30.slcio",
        "/ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MAIA_v0/v8/special/nuGun_filtered_30_70.slcio",
        "/ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MAIA_v0/v8/special/nuGun_filtered_70_110.slcio",
        "/ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MAIA_v0/v8/special/nuGun_filtered_110_150.slcio",
        "/ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MAIA_v0/v8/special/nuGun_filtered_150_180.slcio",
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


def collect_tracks(file_list, label):
    """Loop over all files and events; return (track_arrays, layer_hits).

    layer_hits: {sys_id: {layer: total_hit_count_across_all_tracks}}
    Divide by n_tracks to get average hits per track per layer.
    """
    arrays = {
        "vxd_barrel_nhits": [], "vxd_endcap_nhits": [],
        "it_barrel_nhits":  [], "it_endcap_nhits":  [],
        "ot_barrel_nhits":  [], "ot_endcap_nhits":  [],
        "total_nhits":  [],
        "eta":          [],
        "phi":          [],
        "chi2_reduced": [],
        "d0":           [],
        "z0":           [],
        "pt":           [],
    }
    layer_hits = {sys_id: defaultdict(int) for sys_id in SYS_TO_KEY}

    for fname in file_list:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[{label}] reading {fname}  (RSS so far: {rss_mb:.0f} MB)")
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)

        for ievt, event in enumerate(reader):
            if ievt % 500 == 0:
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"  [{label}] event {ievt}, tracks so far: {len(arrays['total_nhits'])}, RSS: {rss_mb:.0f} MB")
            encoding = get_encoding(event)
            if encoding is None:
                print(f"  event {ievt}: no hit collection found, skipping")
                continue
            decoder = UTIL.BitField64(encoding)

            try:
                track_col = event.getCollection(TRACK_COLLECTION)
            except Exception:
                print(f"  event {ievt}: '{TRACK_COLLECTION}' not found, skipping")
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

                for k, v in counters.items():
                    arrays[k].append(v)
                arrays["total_nhits"].append(sum(counters.values()))
                arrays["eta"].append(eta_from_tanl(track.getTanLambda()))
                arrays["phi"].append(track.getPhi())
                arrays["chi2_reduced"].append(track.getChi2() / float(ndf))
                arrays["d0"].append(track.getD0())
                arrays["z0"].append(track.getZ0())
                omega = track.getOmega()
                arrays["pt"].append(0.3 * BFIELD / (abs(omega) * 1e3) if omega != 0.0 else np.nan)

    return {k: np.array(v) for k, v in arrays.items()}, layer_hits


# ── Data collection ────────────────────────────────────────────────────────

all_data = []   # list of (label, data_dict, layer_hits_dict)
for label, file_list in TRACK_SETS:
    data, layer_hits = collect_tracks(file_list, label)
    all_data.append((label, data, layer_hits))
    print(f"{label}: {len(data['total_nhits'])} tracks")

print()

# ── Plotting helpers ───────────────────────────────────────────────────────

def overlay_hist(ax, datasets, bins, xlabel, title, xscale="linear", yscale="linear", yunit=""):
    """datasets: list of (values_array, label) pairs."""
    kw = dict(histtype="step", linewidth=1.5)
    for values, label in datasets:
        if len(values):
            ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), label=label, **kw)
    ax.set_xlabel(xlabel)
    bw = float(np.diff(bins).mean())
    bw_str = f"{bw:g}" if bw == int(bw) else f"{bw:.3g}"
    unit_str = f" {yunit}" if yunit else ""
    ax.set_ylabel(f"Fraction of tracks / ({bw_str}{unit_str})")
    ax.set_title(title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)


def make_datasets(key):
    return [(data[key], label) for label, data, _ in all_data]


# (key, bins, yunit, xlabel, title, xscale, yscale, filename)
PLOT_SPECS = [
    # ── subdetector hit counts ──────────────────────────────────────────────
    ("vxd_barrel_nhits", np.arange(0, 9),          "hit",  "VXD barrel hits",  "VXD Barrel Hits per Track",  "linear", "linear", "vxd_barrel_hits.pdf"),
    ("vxd_endcap_nhits", np.arange(0, 9),          "hit",  "VXD endcap hits",  "VXD Endcap Hits per Track",  "linear", "linear", "vxd_endcap_hits.pdf"),
    ("it_barrel_nhits",  np.arange(0, 6),          "hit",  "IT barrel hits",   "IT Barrel Hits per Track",   "linear", "linear", "it_barrel_hits.pdf"),
    ("it_endcap_nhits",  np.arange(0, 9),          "hit",  "IT endcap hits",   "IT Endcap Hits per Track",   "linear", "linear", "it_endcap_hits.pdf"),
    ("ot_barrel_nhits",  np.arange(0, 5),          "hit",  "OT barrel hits",   "OT Barrel Hits per Track",   "linear", "linear", "ot_barrel_hits.pdf"),
    ("ot_endcap_nhits",  np.arange(0, 6),          "hit",  "OT endcap hits",   "OT Endcap Hits per Track",   "linear", "linear", "ot_endcap_hits.pdf"),
    # ── totals & track parameters ───────────────────────────────────────────
    ("total_nhits",      np.arange(0, 25),         "hit",  "Total hits",       "Total Hits per Track",       "linear", "linear", "total_hits.pdf"),
    ("eta",              np.linspace(-3, 3, 61),   "",     "Track η",          "Track η",                    "linear", "linear", "eta.pdf"),
    ("phi",              np.linspace(-pi, pi, 65), "rad",  "Track φ [rad]",    "Track φ",                    "linear", "linear", "phi.pdf"),
    ("chi2_reduced",     np.linspace(0, 5, 6),     "",     "χ²/Ndf",           "Track χ²/Ndf",               "linear", "linear", "chi2_reduced.pdf"),
    ("d0",               np.linspace(-5, 5, 51),   "mm",   "d₀ [mm]",          "Track d₀",                   "linear", "linear", "d0.pdf"),
    ("z0",               np.linspace(-25, 25, 101),"mm",   "z₀ [mm]",          "Track z₀",                   "linear", "linear", "z0.pdf"),
    ("pt",               np.logspace(-1, 4, 61),  "GeV",  "Track $p_T$ [GeV]", "Track $p_T$",                "log",    "log",    "pt.pdf"),
]

# ── Individual PDFs ────────────────────────────────────────────────────────

for key, bins, yunit, xlabel, title, xscale, yscale, fname in PLOT_SPECS:
    fig, ax = plt.subplots(figsize=(6, 4))
    overlay_hist(ax, make_datasets(key), bins, xlabel, title, xscale, yscale, yunit=yunit)
    if key == "pt":
        stats_lines = []
        for label, data, _ in all_data:
            pt = data["pt"][np.isfinite(data["pt"])]
            stats_lines.append(f"{label}: median={np.median(pt):.3g} GeV, mean={np.mean(pt):.3g} GeV")
        ax.text(0.03, 0.03, "\n".join(stats_lines), transform=ax.transAxes, fontsize=7.5,
                va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ── Combined multi-page PDF ────────────────────────────────────────────────

combined_path = os.path.join(OUTPUT_DIR, "all_plots.pdf")
with PdfPages(combined_path) as pdf:

    # Page 1: subdetector hit counts (3×2)
    fig, axs = plt.subplots(3, 2, figsize=(11, 12))
    for ax, (key, bins, yunit, xlabel, title, xscale, yscale, _) in zip(axs.flat, PLOT_SPECS[:6]):
        overlay_hist(ax, make_datasets(key), bins, xlabel, title, xscale, yscale, yunit=yunit)
    fig.suptitle("Subdetector Hit Counts per Track", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: total hits + track parameters (3×2)
    fig, axs = plt.subplots(3, 2, figsize=(11, 12))
    for ax, (key, bins, yunit, xlabel, title, xscale, yscale, _) in zip(axs.flat, PLOT_SPECS[6:12]):
        overlay_hist(ax, make_datasets(key), bins, xlabel, title, xscale, yscale, yunit=yunit)
    fig.suptitle("Total Hits & Track Parameters", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 3: track pT
    fig, ax = plt.subplots(figsize=(7, 5))
    key, bins, yunit, xlabel, title, xscale, yscale, _ = PLOT_SPECS[12]
    overlay_hist(ax, make_datasets(key), bins, xlabel, title, xscale, yscale, yunit=yunit)
    stats_lines = []
    for label, data, _ in all_data:
        pt = data["pt"][np.isfinite(data["pt"])]
        stats_lines.append(f"{label}: median={np.median(pt):.3g} GeV, mean={np.mean(pt):.3g} GeV")
    ax.text(0.03, 0.03, "\n".join(stats_lines), transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.suptitle("Track Transverse Momentum", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"\nCombined plot saved to {combined_path}")

# ── Average hits per track vs. layer number ────────────────────────────────

SYS_NAMES = {
    1: "VXD Barrel", 2: "VXD Endcap",
    3: "IT Barrel",  4: "IT Endcap",
    5: "OT Barrel",  6: "OT Endcap",
}

n_sets = len(all_data)
bar_width = 0.6 / n_sets

layer_pdf_path = os.path.join(OUTPUT_DIR, "avg_hits_per_layer.pdf")
with PdfPages(layer_pdf_path) as pdf:
    fig, axs = plt.subplots(3, 2, figsize=(11, 12))
    for ax, (sys_id, sys_name) in zip(axs.flat, SYS_NAMES.items()):
        all_layers = sorted(set().union(*[lh[sys_id].keys() for _, _, lh in all_data]))
        x = np.array(all_layers)
        for i, (label, data, layer_hits) in enumerate(all_data):
            n_tracks = len(data["total_nhits"])
            lh = layer_hits[sys_id]
            offset = (i - n_sets / 2.0 + 0.5) * bar_width
            avg = np.array([lh.get(l, 0) / n_tracks for l in all_layers]) if n_tracks else np.zeros(len(x))
            ax.bar(x + offset, avg, width=bar_width, label=label, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Avg hits / track")
        ax.set_title(f"{sys_name}")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    fig.suptitle("Average Hits per Track vs. Layer", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Layer plot saved to {layer_pdf_path}")
print("Done.")

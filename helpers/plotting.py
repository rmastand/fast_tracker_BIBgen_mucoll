import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def plot_hists_1d(data_dict, bins_dict, log_dims=[0], labels = None, yscale_log=True):
    N_FEATURES = data_dict[list(data_dict.keys())[0]].shape[1]
    fig, ax = plt.subplots(1, N_FEATURES, figsize=(N_FEATURES*6, N_FEATURES))
    for i in range(N_FEATURES):
        for key in data_dict.keys():
            ax[i].hist(data_dict[key][:,i], bins=bins_dict[i], density=True, histtype = "step", label=key, lw=2)
        if labels is None:
            ax[i].set_xlabel(f"Feature {i}")
        else:
            ax[i].set_xlabel(labels[i])
        if yscale_log:
            ax[i].set_yscale("log")
        if i in log_dims:
            ax[i].set_xscale("log")
    ax[-1].legend(loc=(1,0))
    ax[0].set_ylabel("Density")
    return fig, ax




def plot_corner_hist_2d(
    X,
    feature_labels,
    bins_dict,
    log_dims=(0, 4),
    title=None,
):
    """
    X : (N, D) array
    """
    D = X.shape[1]

    # Pre-compute vmin/vmax so LogNorm is valid even when many bins are empty
    all_pos_vals = []
    for i in range(D):
        for j in range(D):
            if j > i:
                h, _, _ = np.histogram2d(
                    X[:, i], X[:, j],
                    bins=[bins_dict[i], bins_dict[j]],
                    density=True,
                )
                pos = h[h > 0]
                if len(pos):
                    all_pos_vals.extend(pos.tolist())

    if all_pos_vals:
        norm = LogNorm(vmin=min(all_pos_vals), vmax=max(all_pos_vals))
    else:
        norm = None  # fall back to linear if no positive-density bins

    fig, axes = plt.subplots(
        D, D,
        figsize=(2 * D, 2 * D),
        constrained_layout=True,
        sharex="col",
        #sharey="row",
    )

    mappable = None

    for i in range(D):
        for j in range(D):
            ax = axes[j, i]

            if j < i:
                ax.axis("off")
                continue

            if i == j:
                # 1D histogram on the diagonal
                ax.hist(
                    X[:, i],
                    bins=bins_dict[i],
                    density=True,
                    histtype = "step",
                )

                if i in log_dims:
                    ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_yticks([])

            else:
                h = ax.hist2d(
                    X[:, i],
                    X[:, j],
                    bins=[bins_dict[i], bins_dict[j]],
                    norm=norm,
                    density=True,
                )
                mappable = h[3]

                if i in log_dims:
                    ax.set_xscale("log")
                if j in log_dims:
                    ax.set_yscale("log")

            # y ticks only on left column
            if i != 0:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            
            # x ticks only on bottom row
            if j != D - 1:
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            

            # labels
            if j == D - 1:
                ax.set_xlabel(feature_labels[i])
            if i == 0 and j != i:
                ax.set_ylabel(feature_labels[j])

    if title is not None:
        fig.suptitle(title, fontsize=16)

    if mappable is not None and norm is not None:
        fig.colorbar(
            mappable,
            ax=axes,
            label="density",
            shrink=0.8,
        )

    #plt.subplots_adjust(hspace=0, wspace=0)

    return fig, axes


def make_2d_plots(data_array, labels, title):


    fig, ax = plt.subplots(1, len(data_array), figsize = (10*len(data_array), 10))

    

    for i, key in enumerate(data_array.keys()):

        coord0_lim = np.min(data_array["Sim BIB"][0]), np.max(data_array["Sim BIB"][0])
        coord1_lim = np.min(data_array["Sim BIB"][1]), np.max(data_array["Sim BIB"][1])
            
        ax[i].scatter(data_array[key][0], data_array[key][1], s = 0.001)
        ax[i].set_xlim(coord0_lim)
        ax[i].set_ylim(coord1_lim)
        ax[i].set_xlabel(labels[0])
        ax[i].set_ylabel(labels[1])
        ax[i].set_title(key)

    plt.suptitle(title)

    return fig, ax


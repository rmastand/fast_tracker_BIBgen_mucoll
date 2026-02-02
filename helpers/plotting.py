import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def plot_hists_1d(data_dict, bins_dict):
    N_FEATURES = len(bins_dict.keys())
    fig, ax = plt.subplots(1, N_FEATURES, figsize=(N_FEATURES*5, N_FEATURES))
    for i in range(N_FEATURES):
        for key in data_dict.keys():
            ax[i].hist(data_dict[key][:,i], bins=bins_dict[i], density=True, histtype = "step", label = key)
        ax[i].set_title(f"Feature {i}")
        ax[i].set_yscale("log")
    ax[-1].legend()
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
    norm = LogNorm()

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

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=axes,
            label="density",
            shrink=0.8,
        )

    #plt.subplots_adjust(hspace=0, wspace=0)

    return fig, axes
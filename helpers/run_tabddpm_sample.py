"""
Subprocess entry-point for one round of tabddpm sampling.

Called by 02_generate_samples.py once per rejection-sampling round so that
the large model, diffusion object, and dataset are fully released (process
exit) between rounds, preventing inter-round OOM accumulation.
"""
import argparse
import pickle
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabddpm-root", required=True)
    parser.add_argument("--tabddpm-scripts", required=True)
    parser.add_argument("--kwargs-path", required=True,
                        help="Path to pickled call kwargs (excluding y_to_sample)")
    parser.add_argument("--y-path", default=None,
                        help="Path to y_to_sample .npy file; omit for Y_MODE=none")
    opts = parser.parse_args()

    for p in (opts.tabddpm_root, opts.tabddpm_scripts):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from sample import sample as tabddpm_sample

    with open(opts.kwargs_path, "rb") as f:
        kwargs = pickle.load(f)

    # device was stored as a string to avoid torch.device pickle issues
    kwargs["device"] = torch.device(kwargs["device"])
    kwargs["y_to_sample"] = np.load(opts.y_path) if opts.y_path else None

    tabddpm_sample(**kwargs)


if __name__ == "__main__":
    main()

import numpy as np
import torch

def sample_from_flow(flow, N=None, x_context=None):
    flow.eval()
    with torch.no_grad():
        if x_context is not None:
            samples = flow(x_context).sample().detach().cpu().numpy()
            samples = np.hstack([
                        samples,
                        x_context.cpu().numpy()])
        else:
            samples = flow().sample((N,)).detach().cpu().numpy()

        return samples


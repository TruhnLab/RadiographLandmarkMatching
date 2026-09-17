import torch


def kde(x, std = 0.1, half = False, down = None, chunk_size = 4096):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    y = x[::down] if down is not None else x
    # Row-chunked with in-place ops: the same per-element arithmetic as
    # (-cdist(x, y)**2 / (2*std**2)).exp().sum(-1), but without allocating several
    # full N x N temporaries (N = 40k matches -> 6.4 GB each in float32).
    density = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(0, x.shape[0], chunk_size):
        scores = torch.cdist(x[i:i + chunk_size], y)
        scores.pow_(2).neg_().div_(2 * std**2).exp_()
        density[i:i + chunk_size] = scores.sum(dim=-1)
    return density

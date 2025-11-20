import torch
import numpy as np
from scipy.optimize import differential_evolution
import torch.nn.functional as F

def get_loss_fn(name: str, temp: float = 0.5):
    """
    Return a function loss_fn(emb_adv, emb_clean) -> scalar torch.Tensor
    The returned value is a *measure we want to maximize* (higher = better adversarial objective).

    Supported names: 'mse', 'kl', 'cosine', 'l2' (euclidean distance on embeddings)
    """
    name = name.lower()

    if name == "mse":
        def fn(emb_adv, emb_clean):
            return F.mse_loss(emb_adv, emb_clean, reduction='mean')
        return fn

    if name == "l2":
        def fn(emb_adv, emb_clean):
            # mean euclidean distance across batch
            return torch.norm(emb_adv - emb_clean, dim=-1).mean()
        return fn

    if name == "kl":
        def fn(emb_adv, emb_clean):
            # KL divergence between clean distribution and adv distribution (maximize)
            p = F.log_softmax(emb_clean / temp, dim=-1)
            q = F.softmax(emb_adv / temp, dim=-1)
            # use batchmean for stability
            return F.kl_div(p, q, reduction='batchmean')
        return fn

    if name == "cosine":
        def fn(emb_adv, emb_clean):
            # cosine similarity (we want to *minimize* similarity, so we maximize its negative later)
            sim = F.cosine_similarity(emb_adv, emb_clean, dim=-1)
            return sim.mean()
        return fn

    raise ValueError(f"Unknown loss name: {name}. Choose from mse, kl, cosine, l2.")


def one_pixel_attack(model, x_with_path, temp=0.5, maxiter=100, popsize=30, loss = "l2"):

    assert loss in ["mse", "kl", "l2", "cosine"]

    x, pth = x_with_path
    loss_fn = get_loss_fn(loss)

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    B, _, H, W = x.shape

    def objective_function(candidate):
        
        x_adv = x.clone()
        for i in range(B):
            start = i * 5
            xi, yi = int(round(candidate[start])), int(round(candidate[start + 1]))
            r, g, b = candidate[start + 2:start + 5]
            r = float(np.clip(r, 0.0, 1.0))
            g = float(np.clip(g, 0.0, 1.0))
            b = float(np.clip(b, 0.0, 1.0))
            xi = max(0, min(H - 1, xi))
            yi = max(0, min(W - 1, yi))
            x_adv[i, 0, xi, yi] = r
            x_adv[i, 1, xi, yi] = g
            x_adv[i, 2, xi, yi] = b

        emb_adv = model.get_embeddings(x_adv)
        
        # we want to maximize the measure, but differential_evolution minimizes the objective,
        # so return negative measure here (minimize -measure == maximize measure)
        measure = loss_fn(emb_adv, emb_clean)
        return -measure.item()

    bounds = []
    for i in range(B):
        bounds += [(0, H - 1), (0, W - 1), (0, 1), (0, 1), (0, 1)]

    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-5,
        workers=1
    )

    # apply best candidate
    x_adv = x.clone()
    for i in range(B):
        cand = result.x[i*5:(i+1)*5]
        xi, yi = int(round(cand[0])), int(round(cand[1]))
        color = torch.tensor(cand[2:5], dtype=x.dtype, device=x.device).view(3, 1, 1)
        xi = max(0, min(H - 1, xi))
        yi = max(0, min(W - 1, yi))
        x_adv[i, :, xi, yi] = color.squeeze()

    return x_adv

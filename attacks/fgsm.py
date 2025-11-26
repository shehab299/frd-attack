import torch.nn.functional as F
import torch
from torch_dct import dct_2d, idct_2d

import torch
import torch.nn.functional as F

def fgsm_l2_attack(model, x, epsilon):
    """
    FGSM attack to maximize L2 distance between original and adversarial embeddings.
    """
    # Unpack input
    x, pth = x

    # Make a fresh copy of x for gradient computation
    # Initialize with small random noise to ensure non-zero gradient
    noise = torch.randn_like(x) * 1e-4
    x_adv = (x + noise).clamp(0, 1).detach().requires_grad_(True)

    # Clean embeddings (no grad)
    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    # Forward pass for adversarial embeddings (keeps grad)
    emb_adv = model.get_embeddings(x_adv)

    # L2 loss to maximize
    loss = torch.norm(emb_adv - emb_clean)
    loss.backward()

    # FGSM step
    x_adv = x_adv + epsilon * x_adv.grad.sign()

    # Clip & detach so next iteration doesn't accumulate computation graph
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv, pth

def fgsm_l2_attack_dct(model, x, epsilon, temp=0.5, save_dir=None):
    """
    FGSM attack to maximize KL divergence between original and adversarial embeddings.

    Args:
        model: embedding model with get_embeddings().
        x: (img_tensor, path) tuple.
        epsilon: float L_inf bound.
        temp: temperature for KL softmax.
    """
    # unpack
    img, path = x  

    # Step 1: move to DCT domain
    # Initialize with small random noise
    x_dct = dct_2d(img)
    noise = torch.randn_like(x_dct) * 1e-4
    x_adv_dct = (x_dct + noise).detach().requires_grad_(True)

    # Reconstruct pixels
    x_adv_pixel = idct_2d(x_adv_dct)

    with torch.no_grad():
        emb_clean = model.get_embeddings(img)  # [B, D]

    emb_adv = model.get_embeddings(x_adv_pixel) # [B, D]

    loss = torch.norm(emb_adv - emb_clean)
    loss.backward()

    # Extract gradient and detach from computation graph
    grad_sign = x_adv_dct.grad.sign().detach()

    del loss, emb_adv, x_adv_pixel
    torch.cuda.empty_cache()

    with torch.no_grad():
        x_adv_dct = x_adv_dct.detach() + epsilon * grad_sign
        x_adv_pixel = torch.clamp(idct_2d(x_adv_dct), 0, 1)

    return x_adv_pixel, path

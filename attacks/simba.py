import torch
import torch.nn.functional as F
import numpy as np
from torch_dct import dct_2d, idct_2d

def ifft2d(img_fft):
    return np.fft.ifft2(img_fft).real  # Real part only

def apply_channelwise(x_np, fn):
    return np.stack([fn(c) for c in x_np], axis=0)

def compute_kl_from_logits(logits_p, logits_q, T):
    """
    Asymmetric KL(p || q) where:
        p_log = log_softmax(logits_p / T)
        q_log = log_softmax(logits_q / T)
        p_prob = softmax(logits_p / T)
    """
    logits_p_t = torch.as_tensor(logits_p, dtype=torch.float32)
    logits_q_t = torch.as_tensor(logits_q, dtype=torch.float32)

    p_log = F.log_softmax(logits_p_t / T, dim=-1)
    q_log = F.log_softmax(logits_q_t / T, dim=-1)
    p_prob = p_log.exp()

    kl = (p_prob * (p_log - q_log)).sum().item()
    return kl


def simba_attack(
    x_tensor,
    model,
    num_iters=1000,
    epsilon=8/255,
    step_size=0.005,
    T=1.0,
    metric="kl",            # 'l2' or 'kl'
    space="image",          # 'image', 'dct', or 'fft'
    maximize=True,          # True: increase distance, False: decrease
    device="cuda"
):
    """
    SIMBA-style attack.

    Args:
        x_tensor: torch.Tensor (C, H, W), in [0,1], already on `device`.
        model: callable -> given (B, C, H, W) returns raw logits or embeddings.
        metric: 'l2' or 'kl' (KL is asymmetric: KL(p||q)).
        space: perturbation space: 'image', 'dct', or 'fft'.
    """
    assert metric in ["l2", "kl"]
    assert space in ["image", "dct", "fft"]
    x_tensor, pth = x_tensor
    
    x_np = x_tensor.detach().cpu().numpy().astype(np.float32)
    _,C, H, W = x_np.shape

    # Space transforms
    if space == "image":
        to_space = lambda x: x
        from_space = lambda x: np.clip(x, 0, 1)
    elif space == "dct":
        to_space = lambda x: apply_channelwise(x, dct_2d)
        from_space = lambda x: np.clip(apply_channelwise(x, idct_2d), 0, 1)
    elif space == "fft":
        to_space = lambda x: apply_channelwise(x, np.fft.fft2)
        from_space = lambda x: np.clip(apply_channelwise(x, ifft2d), 0, 1)

    # Transform to working domain
    working_data = to_space(x_np)
    working_flat = working_data.flatten()
    n_dims = working_flat.shape[0]
    perm = torch.randperm(n_dims)

    # Benign output
    with torch.no_grad():
        benign_out = model.get_embeddings(x_tensor)
        if isinstance(benign_out, torch.Tensor):
            benign_out = benign_out.cpu().numpy()

    # Initialize score
    if metric == "kl":
        benign_logits = benign_out  # store raw logits for p
        current_score = compute_kl_from_logits(benign_logits, benign_logits, T)  # should be 0
    else:
        benign_embed = benign_out
        current_score = 0.0

    # Main loop
    for i in range(num_iters):
        idx = perm[i % n_dims].item()

        # Create plus & minus candidates
        candidates = []
        for direction in [+1, -1]:
            mod_flat = working_flat.copy()
            mod_flat[idx] += direction * step_size

            if space == "image":
                mod_flat = np.clip(mod_flat, 0, 1)
            elif space in ["dct", "fft"]:
                mod_flat[idx] = np.clip(mod_flat[idx], -epsilon, epsilon)

            mod_img = mod_flat.reshape(C, H, W)
            img_reconstructed = from_space(mod_img)
            candidates.append(torch.tensor(img_reconstructed, dtype=x_tensor.dtype, device=device).unsqueeze(0))
            
        # Forward pass for both candidates
        with torch.no_grad():
            outs = model.get_embeddings(torch.cat(candidates, dim=0))
            if isinstance(outs, torch.Tensor):
                outs = outs.cpu().numpy()

        # Compute scores
        if metric == "kl":
            scores = [compute_kl_from_logits(benign_logits, o, T) for o in outs]
        else:
            scores = [np.linalg.norm(benign_embed - o) for o in outs]

        # Accept change
        if maximize:
            better = np.argmax(scores)
            if scores[better] > current_score:
                working_flat[idx] += (+1 if better == 0 else -1) * step_size
                current_score = scores[better]
        else:
            better = np.argmin(scores)
            if scores[better] < current_score:
                working_flat[idx] += (+1 if better == 0 else -1) * step_size
                current_score = scores[better]

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{num_iters}] Score: {current_score:.4f}")

    # Final adversarial image
    final_img = from_space(working_flat.reshape(C, H, W))
    final_tensor = torch.tensor(final_img, dtype=x_tensor.dtype, device=device)
    return final_tensor


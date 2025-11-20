import torch
import torch.nn.functional as F
from torch_dct import dct_2d, idct_2d

def pgd_l2_attack(model, x, epsilon, alpha=0.01, steps=40):
    x, pth = x
    x_adv = x.clone().detach()
    
    with torch.no_grad():
        emb_clean = model.get_embeddings(x).detach()  # [B, D]
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        # Forward pass
        emb_adv = model.get_embeddings(x_adv)  # [B, D]
        loss = -1 * F.mse_loss(emb_adv, emb_clean)
        print(loss)
        
        # Backward pass
        loss.backward()
        
        # Update with gradient
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            
            # Update
            x_adv = x_adv.detach() + alpha * grad_sign
            
            # Projection
            perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + perturbation, 0.0, 1.0)
        
        # Clear gradients explicitly
        del emb_adv, loss
        torch.cuda.empty_cache()  
    
    return x_adv.detach(), pth

def pgd_l2_attack_dct(model, x, epsilon, alpha=0.01, steps=40):
    
    x, pth = x
    x_adv = dct_2d(x)  # Use dct_2d for 2D images
     
    with torch.no_grad():
        emb_clean = model.get_embeddings(x)  # [B, D]

    for _ in range(steps):

        x_adv.requires_grad_(True) 
        x_adv_pixel = idct_2d(x_adv)  # Use idct_2d for 2D

        emb_adv = model.get_embeddings(x_adv_pixel) # [B, D]

        # Apply temperature scaling
        loss = -1 * F.mse_loss(emb_adv, emb_clean)
        print(loss)
        loss.backward()
        
        with torch.no_grad():
            # Update in dct domain
            x_adv = x_adv + alpha * torch.sgn(x_adv.grad)
                
            # Project back to valid constraints
            x_adv_pixel_new = idct_2d(x_adv)
            perturbation = torch.clamp(x_adv_pixel_new - x, min=-epsilon, max=epsilon)
            x_adv_pixel_clipped = torch.clamp(x + perturbation, min=0.0, max=1.0)
            
            # Convert back to dct domain for next iteration
            x_adv = dct_2d(x_adv_pixel_clipped)

        # Clear gradients for next iteration
        if x_adv.grad is not None:
            x_adv.grad = None

    # Convert final result back to pixel domain
    x_adv_final = idct_2d(x_adv)
    x_adv_final = x_adv_final.detach()
    
    return x_adv_final, pth
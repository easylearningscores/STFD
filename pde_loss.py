import torch
import torch.nn.functional as F

def compute_pde_loss(u, dt, dx, dy, alpha=1.0):
    du_dt = (u[:, 1:, :, :, :] - u[:, :-1, :, :, :]) / dt
    d2u_dx2 = (u[:, :, :, :, :-2] - 2*u[:, :, :, :, 1:-1] + u[:, :, :, :, 2:]) / (dx**2)
    d2u_dy2 = (u[:, :, :, :-2, :] - 2*u[:, :, :, 1:-1, :] + u[:, :, :, 2:, :]) / (dy**2)
    laplacian_u = d2u_dx2 + d2u_dy2
    pde_residual = du_dt - alpha * laplacian_u[:, :-1, :, 1:-1, 1:-1]
    pde_loss = torch.mean(pde_residual**2)
    
    return pde_loss

def compute_loss(pred, target, dt, dx, dy, alpha=1.0):
    mse_loss = F.mse_loss(pred, target)
    pde_loss = compute_pde_loss(pred, dt, dx, dy, alpha)
    total_loss = mse_loss + pde_loss
    
    return total_loss, mse_loss, pde_loss

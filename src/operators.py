import numpy as np

def grad_xy(u):
    gx = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy = np.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy

def divergence(px, py):
    d = np.zeros_like(px)
    d[:, 0] += px[:, 0]
    d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    d[:, -1] -= px[:, -2]
    
    d[0, :] += py[0, :]
    d[1:-1, :] += py[1:-1, :] - py[:-2, :]
    d[-1, :] -= py[-2, :]
    return d

def laplacian(u):
    return divergence(*grad_xy(u))

def energy(I, noisy, lam):
    gx, gy = grad_xy(I)
    return 0.5 * float(np.sum((I - noisy) ** 2)) + 0.5 * lam * float(np.sum(gx * gx + gy * gy))

def energy_grad(I, noisy, lam):
    return (I - noisy) - lam * laplacian(I)
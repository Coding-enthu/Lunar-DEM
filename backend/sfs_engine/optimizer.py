import numpy as np
from scipy.optimize import minimize
from backend.sfs_engine.photometric_model import compute_gradients, compute_normals, render_image_from_normals
import matplotlib.pyplot as plt

def compute_smoothness(z):
    # Pad to maintain shape
    dz_dx = np.diff(z, axis=1, append=z[:, -1:])
    dz_dy = np.diff(z, axis=0, append=z[-1:, :])
    smoothness = np.sum(dz_dx ** 2) + np.sum(dz_dy ** 2)
    return smoothness

import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def optimize_heightmap(observed_img, light_direction, max_iter=200, lambda_smooth=0.05):
    H, W = observed_img.shape
    z0 = 0.01 * np.random.randn(H, W).astype(np.float32)
    z0_flat = z0.flatten()

    iter_counter = {'i': 0}

    # 🔧 Ensure debug folder exists
    ensure_dir("outputs/debug")

    def loss_func(z_flat):
        z = z_flat.reshape(H, W)

        # Photometric Loss
        dz_dx, dz_dy = compute_gradients(z)
        normals = compute_normals(dz_dx, dz_dy)
        simulated = render_image_from_normals(normals, light_direction)
        photometric_loss = np.mean((observed_img - simulated) ** 2)

        # Smoothness Loss
        smoothness_loss = compute_smoothness(z)

        # Total Loss
        total_loss = photometric_loss + lambda_smooth * smoothness_loss

        # 🔍 Debug visualization every 1000 iterations
        if iter_counter['i'] % 1000 == 0:
            print(f"[Iter {iter_counter['i']}] Total Loss: {total_loss:.6f}, Photometric: {photometric_loss:.6f}, Smoothness: {smoothness_loss:.6f}")
            print("Z range during iteration:", np.min(z), np.max(z))

            plt.imshow(z, cmap='gray')
            plt.title(f"DEM Iter {iter_counter['i']}")
            plt.axis('off')
            plt.savefig(f"outputs/debug/dem_iter_{iter_counter['i']:05d}.png")
            plt.close()

        iter_counter['i'] += 1
        return total_loss

    result = minimize(loss_func, z0_flat, method='L-BFGS-B', options={'maxiter': max_iter})

    if not result.success:
        print("Optimization failed:", result.message)
    else:
        print("Optimization successful. Final loss:", result.fun)

    z_optimized = result.x.reshape(H, W)
    return z_optimized

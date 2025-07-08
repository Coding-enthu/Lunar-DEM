import numpy as np
from scipy.optimize import minimize
from backend.sfs_engine.photometric_model import compute_gradients, compute_normals, render_image_from_normals
import matplotlib.pyplot as plt
import os

def compute_smoothness(z):
    dz_dx = np.diff(z, axis=1)
    dz_dy = np.diff(z, axis=0)
    return np.sum(dz_dx ** 2) + np.sum(dz_dy ** 2)

def optimize_heightmap(observed_img, light_direction, max_iter=15000, lambda_smooth=0.05):
    H, W = observed_img.shape
    z0 = (observed_img - np.mean(observed_img)) * 0.1  # better initial guess
    z0_flat = z0.flatten()

    iter_counter = {'i': 0}

    # Create debug directory if it doesn't exist
    os.makedirs("outputs/debug", exist_ok=True)

    def loss_func(z_flat):
        z = z_flat.reshape(H, W)

        dz_dx, dz_dy = compute_gradients(z)
        normals = compute_normals(dz_dx, dz_dy)
        simulated = render_image_from_normals(normals, light_direction)

        photometric_loss = np.mean((observed_img - simulated) ** 2)
        smoothness_loss = compute_smoothness(z)
        total_loss = photometric_loss + lambda_smooth * smoothness_loss

        if iter_counter['i'] % 100 == 0:
            z_min, z_max = z.min(), z.max()
            print(f"[Iter {iter_counter['i']}] Total Loss: {total_loss:.6f}, Photometric: {photometric_loss:.6f}, Smoothness: {smoothness_loss:.6f}")
            print(f"Z range during iteration: {z_min} {z_max}")

            # Debug snapshot
            plt.imshow(z, cmap='gray')
            plt.title(f"Z Iteration {iter_counter['i']}")
            plt.axis('off')
            plt.savefig(f"outputs/debug/dem_iter_{iter_counter['i']:05d}.png")
            plt.close()

        iter_counter['i'] += 1
        return total_loss

    result = minimize(
        loss_func,
        z0_flat,
        method='L-BFGS-B',
        options={
            'maxiter': max_iter,
            'maxfun': 500000,
            'disp': True
        }
    )

    if not result.success:
        print("❌ Optimization failed:", result.message)
    else:
        print("✅ Optimization successful. Final loss:", result.fun)

    return result.x.reshape(H, W)

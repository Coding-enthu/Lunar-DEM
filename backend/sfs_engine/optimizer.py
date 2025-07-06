import numpy as np
from scipy.optimize import minimize
from backend.sfs_engine.photometric_model import compute_gradients, compute_normals, render_image_from_normals


def optimize_heightmap(observed_img, light_direction, max_iter=200):
    H, W = observed_img.shape

    # Initial guess for z (height map): start with flat surface
    z0 = np.zeros((H, W), dtype=np.float32)

    # Flatten it to 1D for the optimizer
    z0_flat = z0.flatten()

    # Cost function
    def loss_func(z_flat):
        z = z_flat.reshape(H, W)

        # Compute gradients
        dz_dx, dz_dy = compute_gradients(z)
        normals = compute_normals(dz_dx, dz_dy)
        simulated = render_image_from_normals(normals, light_direction)

        # Compute mean squared error
        error = observed_img - simulated
        loss = np.mean(error ** 2)
        return loss

    # Run optimization using L-BFGS-B (efficient for large problems)
    result = minimize(loss_func, z0_flat, method='L-BFGS-B', options={'maxiter': max_iter})

    if not result.success:
        print("Optimization failed:", result.message)

    # Reshape the result back into a 2D heightmap
    z_optimized = result.x.reshape(H, W)
    return z_optimized

import numpy as np
from scipy.ndimage import zoom
from backend.sfs_engine.optimizer import optimize_heightmap

def build_image_pyramid(image, levels):
    """
    Downsamples the image to create a pyramid (coarse to fine)
    """
    pyramid = [image]
    for i in range(1, levels):
        scale = 0.5 ** i
        downsampled = zoom(image, zoom=scale, order=1)
        pyramid.append(downsampled)
    return pyramid[::-1]  # Return coarse-to-fine order

def pyramid_sfs_solver(image, light_direction, levels=4, lambda_smooth=0.1):
    """
    Coarse-to-fine Shape-from-Shading using a multiscale image pyramid.
    - At each level: initialize with upsampled DEM from coarser level
    - Refine it using photometric + smoothness loss
    """
    pyramid = build_image_pyramid(image, levels)
    prev_dem = None

    for level, img_level in enumerate(pyramid):
        print(f"\n🌀 Level {level+1}/{levels} — Resolution: {img_level.shape}")

        if prev_dem is None:
            # Start from zero elevation for coarsest
            z_init = np.zeros_like(img_level)
        else:
            # Upsample previous DEM to current resolution
            scale_y = img_level.shape[0] / prev_dem.shape[0]
            scale_x = img_level.shape[1] / prev_dem.shape[1]
            z_init = zoom(prev_dem, zoom=(scale_y, scale_x), order=1)

        # Inject the initialization into the optimizer
        dem_level = optimize_heightmap(
            img_level,
            light_direction,
            lambda_smooth=lambda_smooth,
            z_initial=z_init
        )

        print(f"Z range at level {level+1}: {dem_level.min():.6f} to {dem_level.max():.6f}")
        prev_dem = dem_level

    return prev_dem
# This function returns the final DEM at the finest resolution
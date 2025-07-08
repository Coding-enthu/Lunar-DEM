import numpy as np
import matplotlib.pyplot as plt
import os

from backend.preprocess.image_cleaner import preprocess_image
from backend.visualizer import plot_3d_elevation_map
from backend.sfs_engine.optimizer import optimize_heightmap
from backend.sfs_engine.pyramid_solver import pyramid_sfs_solver
from backend.sfs_engine.photometric_model import (
    compute_gradients,
    compute_normals,
    render_image_from_normals
)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_synthetic_test(light_dir, use_pyramid=False):
    print("\n🧪 Running synthetic DEM test...")

    # Generate a smooth synthetic elevation surface
    H, W = 256, 256
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-1, 1, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z_test = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)

    # Generate synthetic shading image from elevation
    dz_dx, dz_dy = compute_gradients(z_test)
    normals = compute_normals(dz_dx, dz_dy)
    sim = render_image_from_normals(normals, light_dir)

    ensure_dir("outputs")
    plt.imsave("outputs/simulated_image.png", sim, cmap='gray')
    plt.imshow(sim, cmap='gray')
    plt.title("Simulated Image from Synthetic DEM")
    plt.show()

    # Recover elevation
    if use_pyramid:
        dem = pyramid_sfs_solver(sim, light_dir, levels=4, lambda_smooth=0.05)
    else:
        dem = optimize_heightmap(sim, light_dir, lambda_smooth=0.05)

    # Evaluate and visualize
    error = np.mean((dem - z_test) ** 2)
    print(f"✅ Reconstruction MSE: {error:.6f}")
    print(f"Z range: {dem.min():.6f} to {dem.max():.6f}")

    plt.imshow(dem, cmap="gray")
    plt.title("Recovered DEM from Synthetic Image")
    plt.colorbar()
    plt.savefig("outputs/recovered_dem_from_sim.png")
    plt.show()

    plot_3d_elevation_map(dem)

def run_real_image(light_dir, use_pyramid=False):
    print("\n🌑 Running real lunar image pipeline...")

    # Load and preprocess image
    img = preprocess_image("images/lunar_sample.png")
    print("Input image stats — min:", img.min(), "max:", img.max())

    plt.imshow(img, cmap='gray')
    plt.title("Preprocessed Input Image")
    ensure_dir("outputs")
    plt.savefig("outputs/preprocessed_lunar_input.png")
    plt.show()

    # Solve DEM
    if use_pyramid:
        dem = pyramid_sfs_solver(img, light_dir, levels=4, lambda_smooth=0.2)
    else:
        dem = optimize_heightmap(img, light_dir, lambda_smooth=0.2)

    # Save elevation map
    plt.imshow(dem, cmap="gray")
    plt.title("Estimated Elevation Map")
    plt.colorbar()
    plt.savefig("outputs/elevation_map.png")
    plt.show()

    plot_3d_elevation_map(dem)

if __name__ == "__main__":
    # Define light direction and normalize
    light_dir = np.array([0.6, -0.4, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # ✅ Toggle these to switch behavior
    use_synthetic = False         # ⬅️ Set to False for real lunar image
    use_pyramid_solver = True     # ⬅️ Set to True to use pyramid optimization

    if use_synthetic:
        run_synthetic_test(light_dir, use_pyramid=use_pyramid_solver)
    else:
        run_real_image(light_dir, use_pyramid=use_pyramid_solver)

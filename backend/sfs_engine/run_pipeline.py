import numpy as np
import matplotlib.pyplot as plt
from backend.preprocess.image_cleaner import preprocess_image
from backend.sfs_engine.optimizer import optimize_heightmap
from backend.visualizer import plot_3d_elevation_map
from backend.sfs_engine.photometric_model import (
    compute_gradients,
    compute_normals,
    render_image_from_normals
)
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_synthetic_test(light_dir):
    print("\n🧪 Running synthetic DEM test...")

    # Generate a smooth ripple elevation surface
    H, W = 256, 256
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-1, 1, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z_test = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)

    # Compute synthetic image
    dz_dx, dz_dy = compute_gradients(z_test)
    normals = compute_normals(dz_dx, dz_dy)
    sim = render_image_from_normals(normals, light_dir)

    # Save synthetic image
    ensure_dir("outputs")
    plt.imsave("outputs/simulated_image.png", sim, cmap='gray')

    plt.imshow(sim, cmap='gray')
    plt.title("Simulated Image from Synthetic DEM")
    plt.show()

    # Optimize DEM from simulated image
    dem = optimize_heightmap(sim, light_dir)

    # Evaluate reconstruction error
    error = np.mean((dem - z_test) ** 2)
    print(f"✅ Reconstruction MSE: {error:.6f}")
    print(f"Z range: {dem.min():.4f} to {dem.max():.4f}")

    # Save and show DEM
    plt.imshow(dem, cmap="gray")
    plt.title("Recovered DEM from Synthetic Image")
    plt.colorbar()
    plt.savefig("outputs/recovered_dem_from_sim.png")
    plt.show()

    plot_3d_elevation_map(dem)

def run_real_image(light_dir):
    print("\n🌑 Running real lunar image pipeline...")

    # Load and preprocess
    img = preprocess_image("images/lunar_sample.png")
    print("Input image stats — min:", img.min(), "max:", img.max())
    plt.imshow(img, cmap='gray')
    plt.title("Preprocessed Input Image")
    plt.savefig("outputs/preprocessed_lunar_input.png")
    plt.show()

    # Optimize DEM from real image
    dem = optimize_heightmap(img, light_dir)

    # Save elevation map
    plt.imshow(dem, cmap="gray")
    plt.title("Estimated Elevation Map")
    plt.colorbar()
    plt.savefig("outputs/elevation_map.png")
    plt.show()

    # Plot 3D surface
    plot_3d_elevation_map(dem)

if __name__ == "__main__":
    # Define normalized light source
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Toggle which mode to run
    use_synthetic = False  # ⬅️ Set to False for real lunar image

    if use_synthetic:
        run_synthetic_test(light_dir)
    else:
        run_real_image(light_dir)

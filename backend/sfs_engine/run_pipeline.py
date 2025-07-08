from backend.preprocess.image_cleaner import preprocess_image
from backend.sfs_engine.optimizer import optimize_heightmap
from backend.visualizer import plot_3d_elevation_map
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 1. Load and preprocess image
    img = preprocess_image("images/lunar_sample.png")

    # 2. Define light source direction
    light_dir = np.array([0, 0, 1])

    # 3. Generate elevation map (DEM)
    dem = optimize_heightmap(img, light_dir)

    # 4. Show 2D grayscale elevation map
    plt.imshow(dem, cmap="gray")
    plt.title("Estimated Elevation Map")
    plt.colorbar()
    plt.savefig("outputs/elevation_map.png")
    plt.show()

    # 5. Plot 3D surface view
    plot_3d_elevation_map(dem)

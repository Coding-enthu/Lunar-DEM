from backend.preprocess.image_cleaner import preprocess_image
from backend.sfs_engine.optimizer import optimize_heightmap
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Load image
    img = preprocess_image("images/lunar_sample.png")
    # if not img:
    #     print("###Image cannt be loaded")

    # Define light direction
    light_dir = np.array([0, 0, 1])

    # Compute elevation map
    dem = optimize_heightmap(img, light_dir)

    # Display elevation
    plt.imshow(dem, cmap="gray")
    plt.title("Estimated Lunar Elevation Map")
    plt.colorbar()
    plt.savefig("outputs/elevation_map.png")
    plt.show()

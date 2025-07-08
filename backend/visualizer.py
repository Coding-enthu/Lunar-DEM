import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_elevation_map(dem: np.ndarray, save_path: str = "outputs/elevation_3d.png"):
    """
    Plots a 3D surface view of the given elevation map (DEM).

    Parameters:
        dem (np.ndarray): 2D array representing the elevation map (height at each pixel).
        save_path (str): File path to save the generated 3D image.
    """
    H, W = dem.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, dem, cmap='terrain', edgecolor='none', linewidth=0)
    ax.set_title("3D View of Lunar DEM")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")

    # Optional: adjust camera view
    ax.view_init(elev=60, azim=45)

    plt.savefig(save_path)
    plt.show()

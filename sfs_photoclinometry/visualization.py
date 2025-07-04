import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d(Z, title="Lunar DEM"):
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=True)
    ax.set_title(title)
    plt.show()

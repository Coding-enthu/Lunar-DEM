import numpy as np
import os
import rasterio
from rasterio.transform import from_origin

def save_dem(Z, path="output/dem.npy"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, Z)

def save_geotiff(Z, path="output/dem.tif", resolution=5.0):

    if not isinstance(Z, np.ndarray):
        raise TypeError(f"[ERROR] Z should be a numpy array, got {type(Z)}")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    transform = from_origin(0, 0, resolution, resolution)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=Z.shape[0], width=Z.shape[1],
        count=1, dtype=Z.dtype, transform=transform,
        crs='+proj=latlong'
    ) as dst:
        dst.write(Z, 1)

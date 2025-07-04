from config_manager import load_config, sun_vector
from preprocessing import preprocess_image
from sfs_photoclinometry.core import run_sfs
from sfs_photoclinometry.io_handler import save_dem, save_geotiff
from sfs_photoclinometry.visualization import plot_3d

def main():
    config = load_config()
    image_path = config["image_path"]
    azimuth = config["sun_azimuth"]
    elevation = config["sun_elevation"]
    light_vec = sun_vector(azimuth, elevation)

    image = preprocess_image(image_path)
    Z = run_sfs(image, light_vec, max_iter=config["max_iterations"])

    save_dem(Z)
    save_geotiff(Z)
    plot_3d(Z)

if __name__ == "__main__":
    main()

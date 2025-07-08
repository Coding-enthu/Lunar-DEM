import numpy as np

def compute_gradients(z):
    dz_dy, dz_dx = np.gradient(z.astype(np.float32))

    # Ensure all have the same shape
    dz_dx = dz_dx[:z.shape[0], :z.shape[1]]
    dz_dy = dz_dy[:z.shape[0], :z.shape[1]]

    return dz_dx, dz_dy




def compute_normals(dz_dx, dz_dy):
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(dz_dx)

    normals = np.stack((nx, ny, nz), axis=2)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= (norm + 1e-8)
    return normals

def render_image_from_normals(normals, light_direction):
    shading = np.dot(normals, light_direction)
    shading = np.clip(shading, 0, 1)
    return shading


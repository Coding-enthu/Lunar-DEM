import numpy as np

def compute_gradients(z):
    dz_dx = np.gradient(z, axis=1)  # ∂z/∂x
    dz_dy = np.gradient(z, axis=0)  # ∂z/∂y
    return dz_dx, dz_dy

def compute_normals(dz_dx, dz_dy):
    # Compute surface normals
    normals = np.zeros((*dz_dx.shape, 3))
    normals[..., 0] = -dz_dx
    normals[..., 1] = -dz_dy
    normals[..., 2] = 1.0

    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norm + 1e-8
    return normals

def render_image_from_normals(normals, light_direction):
    # Ensure light direction is a unit vector
    s = light_direction / np.linalg.norm(light_direction)
    
    # Dot product between each normal and the light direction
    intensity = np.dot(normals, s)
    intensity = np.clip(intensity, 0, 1)  # no negative light
    
    return intensity


## Test ##
z = np.random.randn(256, 256) * 0.1  # some random elevation
dz_dx, dz_dy = compute_gradients(z)
normals = compute_normals(dz_dx, dz_dy)

light_dir = np.array([0, 0, 1])  # coming from top
rendered = render_image_from_normals(normals, light_dir)
print(rendered)
import numpy as np

def compute_normals(Z):
    dzdx = np.gradient(Z, axis=1)
    dzdy = np.gradient(Z, axis=0)

    normals = np.stack([-dzdx, -dzdy, np.ones_like(Z)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals/(norm + 1e-6)

def predict_intensity(normals, light_vector):
    return np.tensordot(normals, light_vector, axes=([2], [0]))

def loss_function(Z_flat, image, light_vector, shape):
    Z = Z_flat.reshape(shape)
    normals = compute_normals(Z)
    I_pred = predict_intensity(normals, light_vector)
    error = image - I_pred
    return np.mean(error**2)

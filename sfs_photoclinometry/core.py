import numpy as np
from scipy.optimize import minimize
from .utils import loss_function

def run_sfs(image, light_vector, max_iter=100):
    shape = image.shape
    Z0 = np.zeros(shape, dtype=np.float32).flatten()

    result = minimize(
        loss_function,
        Z0,
        args=(image, light_vector, shape),
        method='L-BFGS-B',
        options={'maxiter': max_iter}
    )
    return result.x.reshape
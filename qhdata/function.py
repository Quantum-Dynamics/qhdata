import numpy as np


def gaussian_dist(
    x: np.ndarray,
    mean: float = 0.,
    std: float = 1.,
) -> np.ndarray:
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(- (x - mean)**2 / std**2)

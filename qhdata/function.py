import numpy as np


def gaussian_dist(
    x: np.ndarray,
    mean: float = 0.,
    std: float = 1.,
) -> np.ndarray:
    r"""Funciton of Gaussian distribution.

    This function returns values of Gaussian distribution:
    ```math
    f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}}
        \exp \left( - \frac{(x - \mu)^2}{\sigma^2} \right)
    ```
    , where $`\mu`$ is a mean and $`\sigma`$ is a standard deviation of the
    distribution.

    Args:
        x: Domain of the function.
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.

    Returns:
        Gaussian distribution with the parameters specified.
    """
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(- (x - mean)**2 / std**2)

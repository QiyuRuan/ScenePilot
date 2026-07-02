import math
import numpy as np


def standard_normal_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal distribution Phi(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def generate_gaussian_epsilon_levels(
    num_levels: int = 8,
    epsilon_max: float = 0.35,
    u_min: float = -1.8,
    u_max: float = 0.0,
) -> np.ndarray:
    """
    Gaussian-scheduled epsilon values.

    Smaller epsilon values are spaced more densely;
    larger epsilon values are spaced more sparsely.
    """
    if num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    if epsilon_max <= 0:
        raise ValueError("epsilon_max must be greater than 0")
    if u_min >= u_max:
        raise ValueError("u_min must be smaller than u_max")

    # Sample uniformly on the truncated standard-normal axis.
    u_values = np.linspace(u_min, u_max, num_levels)

    cdf_min = standard_normal_cdf(u_min)
    cdf_max = standard_normal_cdf(u_max)

    epsilon_levels = np.array([
        epsilon_max
        * (standard_normal_cdf(u) - cdf_min)
        / (cdf_max - cdf_min)
        for u in u_values
    ])

    # Avoid floating-point drift and force endpoints to 0 and epsilon_max.
    epsilon_levels[0] = 0.0
    epsilon_levels[-1] = epsilon_max

    return epsilon_levels


if __name__ == "__main__":
    epsilon_levels = generate_gaussian_epsilon_levels(
        num_levels=8,
        epsilon_max=0.35,
        u_min=-1.8, #-0.5near the mean
        u_max=0.0,
    )

    epsilon_levels = np.round(epsilon_levels, 2)

    print("Gaussian-spaced epsilon levels:")
    for index, epsilon in enumerate(epsilon_levels):
        print(f"epsilon_{index + 1} = {epsilon:.6f}")

    print("\nPython list:")
    print(epsilon_levels.tolist())
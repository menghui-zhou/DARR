import numpy as np


def compute_epsilons(m_list, d):
    """
    Compute ε²_L, ε²_U, and ε²_* given class sample counts and dimension d.

    Args:
        m_list (list or np.array): List of class sample counts [m1, m2, ..., mK]
        d (float): Dimensionality of the feature space

    Returns:
        dict: Dictionary with keys 'eps2_L', 'eps2_U', and 'eps2_star'
    """
    m_list = np.array(m_list)
    m = np.sum(m_list)

    sqrt_ratios = np.sqrt(m_list / m)

    eps2_L = np.sum(sqrt_ratios)
    eps2_U = np.min(d * sqrt_ratios) /3


    return {
        'eps2_L': eps2_L,
        'eps2_U': eps2_U
    }


# Example usage:
m_list = [266, 422]  # sample counts per class
d = 128  # feature dimension
epsilons = compute_epsilons(m_list, d)
print(epsilons)

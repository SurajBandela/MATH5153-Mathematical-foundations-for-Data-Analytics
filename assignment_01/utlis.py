import numpy as np

def create_time_series_data(t):
    """
    Create a time series dataset with noise.

    Parameters:
    time (ndarray): An array of time values.

    Returns:
    ndarray: An array of time series data.
    """
    signal = np.sin(t)
    return signal

def inner_product(f,g, a, b, N, integrator):
    """
    Compute the inner product of two functions f and g over the interval [a,b].

    Parameters:
    f (function): The first function.
    g (function): The second function.
    a (float): The lower bound of the interval.
    b (float): The upper bound of the interval.
    N (int): The number of points to use in the approximation.
    integrator (function): The numerical integration function to use.

    Returns:
    float: The inner product of f and g over the interval [a,b].
    """
    integrand = lambda t: np.multiply(f(t),g(t), dtype=object)
    inner_product = integrator(integrand, a, b, N)
    return inner_product

def create_gram_matrix(basis_list, a, b, integrator):
    """
    Create the Gram matrix for a given set of basis functions.

    Parameters:
    basis_list (list): A list of basis functions.
    a (float): The lower bound of the interval.
    b (float): The upper bound of the interval.
    N (int): The number of points to use in the approximation.
    integrator (function): The numerical integration function to use.

    Returns:
    ndarray: The Gram matrix for the basis functions.
    """
    M = len(basis_list)
    gram_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            f = basis_list[i]
            g = basis_list[j]
            gram_matrix[i, j] = inner_product(f, g, a, b, M, integrator)
    return gram_matrix

def right_hand_vector(basis_list, signal,a,b, integrator):
    """
    Create the right-hand vector for the linear system
    parameters:
    basis_list (list): A list of basis functions.
    signal (function): The signal function.
    a (float): The lower bound of the interval.
    b (float): The upper bound of the interval.
    N (int): The number of points to use in the approximation.
    integrator (function): The numerical integration function to use.

    Returns:
    ndarray: The right-hand vector.
    """
    N = len(basis_list)
    right_hand_vector = np.zeros(N)
    for i in range(N):
        right_hand_vector[i] = inner_product(basis_list[i], signal, a, b, N, integrator)
    return right_hand_vector


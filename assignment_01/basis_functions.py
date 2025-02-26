import numpy as np

def cosine_basis_function(n):
    """
    Returns a cosine basis function: cos(n * t).
    
    Parameters:
        n (int): Frequency parameter of the cosine function.
    
    Returns:
        function: A function `f(t)` representing cos(n*t).
    """
    return lambda t: np.cos(n * t)

def polynomial_exponential_basis_function(n):
    """
    Returns a polynomial exponential basis function: t^n * exp(nt).
    
    Parameters:
        n (int): Degree of the polynomial.
    
    Returns:
        function: A function `f(t)` representing t^n * exp(nt).
    """
    return lambda t:  np.exp(t/n)
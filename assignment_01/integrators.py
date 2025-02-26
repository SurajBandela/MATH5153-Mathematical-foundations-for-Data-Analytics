import math
import numpy as np

def trapezoidal_rule(f, a, b, n):
    """
    Approximate the integral of f from a to b using the trapezoidal rule with n intervals.

    Parameters:
    f (function): The function to integrate.
    a (float): The start point of the interval.
    b (float): The end point of the interval.
    n (int): The number of subintervals.

    Returns:
    float: The approximate integral of f from a to b.
    """
    h = (b - a) / n # width of each subinterval
    x = np.linspace(a,b,n+1)
    y = f(x)

    integral = (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
    return integral



def simpsons_rule(f, a, b, n):
    """
    Approximate the integral of f from a to b using Simpson's rule with n intervals.

    Parameters:
    f (function): The function to integrate.
    a (float): The start point of the interval.
    b (float): The end point of the interval.
    n (int): The number of subintervals (must be even).

    Returns:
    float: The approximate integral of f from a to b.
    """
        
    h = (b - a) / n # width of each subinterval
    x = np.linspace(a,b,n+1)
    y = f(x)

    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])
    integral += 4 * np.sum(y[2:-1:2])
    integral *= h/3
    return integral

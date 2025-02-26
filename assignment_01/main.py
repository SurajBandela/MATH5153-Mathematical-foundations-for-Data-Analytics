import numpy as np
import matplotlib.pyplot as plt
import argparse, os

from integrators import trapezoidal_rule, simpsons_rule
from basis_functions import cosine_basis_function, polynomial_exponential_basis_function
from utlis import create_time_series_data, inner_product, create_gram_matrix, right_hand_vector

def function_approximator(t,c, basis):
    """
    Approximates a function value at a given point using a linear combination of basis functions.

    Parameters:
    t (float): The point at which to evaluate the approximated function.
    c (list of float): Coefficients for the linear combination of basis functions.
    basis (list of callable): List of basis functions, each of which takes a single argument (t).

    Returns:
    float: The approximated function value at point t.
    """
    return sum(c[k] * basis[k](t) for k in range(len(basis)))

parser = argparse.ArgumentParser(description='Function Approximation using Basis Functions')
parser.add_argument('--initial_value', default=-2*np.pi, help='Starting value of the signal (a)')
parser.add_argument('--final_value', default=2*np.pi, help='Final value of the signal (b)')
parser.add_argument('--interval', type=int, default=100, help='Number of intervals (N)')

args = parser.parse_args()

np.random.seed(0)
a = args.initial_value
b = args.final_value
N = args.interval
time = np.linspace(a,b,N)
signal = create_time_series_data(time)

noise = 0.5 * np.random.normal(a,b,N)
noisy_signal = signal + noise

integrator_01 = trapezoidal_rule
basis_01 = cosine_basis_function

integrator_02 = trapezoidal_rule
basis_02 = polynomial_exponential_basis_function 

integrator_03 =  simpsons_rule
basis_03 = cosine_basis_function

integrator_04 =  simpsons_rule
basis_04 = polynomial_exponential_basis_function


basis_list_01 = [basis_01(n) for n in range(1,len(time))]
G_01 = create_gram_matrix(basis_list_01, a, b, integrator_01)
r_01 = right_hand_vector(basis_list_01, lambda t: noisy_signal, a, b, integrator_01)
c_01 = np.linalg.solve(G_01, r_01)
approximator_01 = function_approximator(time,c_01,basis_list_01)

basis_list_02 = [basis_02(n) for n in range(1,len(time))]
G_02 = create_gram_matrix(basis_list_02, a, b, integrator_02)
r_02 = right_hand_vector(basis_list_02, lambda t: noisy_signal, a, b, integrator_02)
c_02 = np.linalg.solve(G_02, r_02)
approximator_02 = function_approximator(time,c_02,  basis_list_02)

basis_list_03 = [basis_03(n) for n in range(1,len(time))]
G_03 = create_gram_matrix(basis_list_03, a, b, integrator_03)
r_03 = right_hand_vector(basis_list_03, lambda t: noisy_signal, a, b, integrator_03)
c_03= np.linalg.solve(G_03, r_03)
approximator_03 = function_approximator(time, c_03, basis_list_03)

basis_list_04 = [basis_04(n) for n in range(1,len(time))]
G_04 = create_gram_matrix(basis_list_04, a, b, integrator_04)
r_04 = right_hand_vector(basis_list_04, lambda t: noisy_signal, a, b, integrator_04)
c_04 = np.linalg.solve(G_04, r_04)
approximator_04 = function_approximator(time, c_04, basis_list_04)

fig_path = os.path.join(os.getcwd(),'assignment_01/plots')
if not os.path.isdir(fig_path):
    os.makedirs(fig_path)

# Plot the time series data
plt.figure(1, figsize=(10, 8))
plt.plot(time, signal, label="Original Function", linestyle="--")
plt.plot(time, noisy_signal, label="Noisy Data", color="red", alpha=0.5)
plt.plot(time, approximator_01 , label="Approximation",color="green")
plt.xlabel("time"), plt.ylabel("f(t)")
plt.title("Function Approximation Using Basis Functions")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_path,'fig_01.pdf'))

plt.figure(2, figsize=(10, 8))
plt.plot(time, signal, label="Original Function", linestyle="--")
plt.plot(time, noisy_signal, label="Noisy Data", color="red", alpha=0.5)
plt.plot(time, approximator_02 , label="Approximation",color="blue")
plt.xlabel("time"), plt.ylabel("f(t)")
plt.title("Function Approximation Using Basis Functions")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_path,'fig_02.pdf'))

plt.figure(3, figsize=(10, 8))
plt.plot(time, signal, label="Original Function", linestyle="--")
plt.plot(time, noisy_signal, label="Noisy Data", color="red", alpha=0.5)
plt.plot(time, approximator_03 , label="Approximation",color="black")
plt.xlabel("time"), plt.ylabel("f(t)")
plt.title("Function Approximation Using Basis Functions")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_path,'fig_03.pdf'))

plt.figure(4, figsize=(10, 8))
plt.plot(time, signal, label="Original Function", linestyle="--")
plt.plot(time, noisy_signal, label="Noisy Data", color="red", alpha=0.5)
plt.plot(time, approximator_04 , label="Approximation",color="orange")
plt.xlabel("time"), plt.ylabel("f(t)")
plt.title("Function Approximation Using Basis Functions")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_path,'fig_04.pdf'))




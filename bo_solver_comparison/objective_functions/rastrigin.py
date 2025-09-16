import numpy as np

# Rastrigin 函数
def rastrigin_function(x, A=10):
    x = np.ravel(x) # 1D
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)) + np.random.normal(loc=0., scale=1.) *1e-2

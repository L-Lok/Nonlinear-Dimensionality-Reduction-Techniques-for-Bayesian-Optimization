import numpy as np

# Rosenbrock 函数
def rosenbrock_function(x, a=1, b=100):
    x = np.ravel(x) # ensure 1d
    N = len(x)
    f = np.sum([b*(x[i + 1] - x[i]**2)**2 + (a - x[i])**2 for i in range(N - 1)])
    return f + np.random.normal(loc=0., scale=1.) *1e-2

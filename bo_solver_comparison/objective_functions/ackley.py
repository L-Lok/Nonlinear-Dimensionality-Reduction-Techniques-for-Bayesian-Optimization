import numpy as np

# Ackley 函数
def ackley_function(x, a=20, b=0.2, c=2 * np.pi):
    x = np.ravel(x) # 1D
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    sum2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum1 + sum2 + a + np.exp(1) + np.random.normal(loc=0., scale=1.)*1e-2

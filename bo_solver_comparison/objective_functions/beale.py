import numpy as np

# Beale 函数
def beale_function(x):
    x = np.ravel(x)
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3 + np.random.normal(loc=0., scale=1.) *1e-2

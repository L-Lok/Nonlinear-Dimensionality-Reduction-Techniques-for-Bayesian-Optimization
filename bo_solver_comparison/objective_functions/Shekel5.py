import numpy as np

# Shekel 5

def shekel5(x):
    """
    Shekel 5 function. Global minimum is f=-10.1499 at x = np.array([4.0, 4.0, 4.0, 4.0])
    :param x: Input vector, length 4
    :return: Float.
    """
    x = np.ravel(x) # ensure 1d
    a1 = np.array([4.0, 4.0, 4.0, 4.0])
    a2 = np.array([1.0, 1.0, 1.0, 1.0])
    a3 = np.array([8.0, 8.0, 8.0, 8.0])
    a4 = np.array([6.0, 6.0, 6.0, 6.0])
    a5 = np.array([3.0, 7.0, 3.0, 7.0])
    a = [a1, a2, a3, a4, a5]
    c = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
    f = -np.sum(np.array([1.0 / (np.sum((x - a[i]) ** 2) + c[i]) for i in range(5)]))
    return f + np.random.normal(loc=0., scale=1.) *1e-2
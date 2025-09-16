import numpy as np

# Hartmann 3

def hartman3(x):
    """
    Hartman 3 function. Global minimum is f=-3.862782 at x = np.array([0.114614, 0.555649, 0.852547])
    :param x: Input vector, length 3
    :return: Float.
    """
    x = np.ravel(x)
    c = np.array([1.0, 1.2, 3.0, 3.2])
    a1 = np.array([3.0, 10.0, 30.0])
    a2 = np.array([0.1, 10.0, 35.0])
    a3 = np.array([3.0, 10.0, 30.0])
    a4 = np.array([0.1, 10.0, 35.0])
    A = [a1, a2, a3, a4]
    p1 = np.array([0.3689, 0.117, 0.2673])
    p2 = np.array([0.4699, 0.4387, 0.747])
    p3 = np.array([0.1091, 0.8732, 0.5547])
    p4 = np.array([0.03815, 0.5743, 0.8828])
    p = [p1, p2, p3, p4]
    # The rest is the same for both Hartman functions
    inner_terms = np.array([- np.dot(A[i], (x-p[i])**2) for i in range(4)])
    f = -np.dot(c, np.exp(inner_terms))
    return f + np.random.normal(loc=0., scale=1.) *1e-2
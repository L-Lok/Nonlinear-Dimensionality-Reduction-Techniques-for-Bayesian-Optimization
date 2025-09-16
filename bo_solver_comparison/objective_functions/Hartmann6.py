import numpy as np

# Hartmann 6 func
def hartman6(x):
    """
    Hartman 6 function. Global minimum is f=-3.322368 at x = np.array([0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301])
    :param x: Input vector, length 6
    :return: Float.
    """
    x = np.ravel(x) # ensure 1d
    c = np.array([1.0, 1.2, 3.0, 3.2])
    a1 = np.array([10.0, 3.0, 17.0, 3.5, 1.7, 8.0])
    a2 = np.array([0.05, 10.0, 17.0, 0.1, 8.0, 14.0])
    a3 = np.array([3.0, 3.5, 1.7, 10.0, 17.0, 8.0])
    a4 = np.array([17.0, 8.0, 0.05, 10.0, 0.1, 14.0])
    A = [a1, a2, a3, a4]
    p1 = np.array([0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886])
    p2 = np.array([0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991])
    p3 = np.array([0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665])
    p4 = np.array([0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381])
    # The rest is the same for both Hartman functions
    p = [p1, p2, p3, p4]
    inner_terms = np.array([- np.dot(A[i], (x - p[i]) ** 2) for i in range(4)])
    f = -np.dot(c, np.exp(inner_terms))
    return f + np.random.normal(loc=0., scale=1.) *1e-2

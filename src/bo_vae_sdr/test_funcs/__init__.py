"""
Test functions
"""

from .Ackley import Ackley
from .BeckerAndLago import BeckerAndLago
from .Hartmann6 import Hartmann6
from .Levy import Levy
from .Pricefunc import PriceTM
from .Rosenbrock import Rosenbrock
from .Shekel5 import Shekel5
from .Sphere import Sphere
from .StyblinskiTang import StyblinskiTang
from .Trid import Trid
from .Rastrigin import Rastrigin

all_funcs = [Ackley, BeckerAndLago, Rosenbrock, Sphere, Shekel5, PriceTM, Hartmann6, StyblinskiTang, Levy,
             Trid, Rastrigin]

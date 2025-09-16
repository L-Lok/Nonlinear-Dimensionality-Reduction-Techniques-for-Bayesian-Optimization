import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


# bayes_opt Wrapper
class BayesOptWrapper:
    def __init__(self, func, bounds, allow_duplicate_points=False):
        pbounds = {f"x{i+1}": b for i, b in enumerate(bounds)}
        self.optimizer = BayesianOptimization(
            f=lambda **kwargs: -func(
                np.array(list(kwargs.values()))
            ),  # for minimisation
            pbounds=pbounds,
            random_state=27,
            allow_duplicate_points=allow_duplicate_points,
        )

    def maximize(self, n_iter=1):
        acq = UtilityFunction(kind="ei")
        best_f = []
        for _ in range(n_iter):
            next_point = self.optimizer.suggest(acq)
            new_y = self.optimizer._space.target_func(**next_point)
            self.optimizer.register(params=next_point, target=new_y)
            best_f.append(
                np.max(
                    [
                        self.optimizer.res[i]["target"]
                        for i in range(len(self.optimizer.res))
                    ]
                )
            )
        return (-1) * np.array(best_f)  # we are doing minimisation

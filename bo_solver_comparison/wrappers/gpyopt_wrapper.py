import GPyOpt
import numpy as np


# GPyOpt Wrapper
class GPyOptWrapper:
    def __init__(self, func, bounds, n_init):
        self.func = func
        self.bounds = [
            {"name": f"var_{i+1}", "type": "continuous", "domain": b}
            for i, b in enumerate(bounds)
        ]

        self.n_init = n_init
        self.optimizer = GPyOpt.methods.BayesianOptimization(
            f=self.func,
            domain=self.bounds,
            model_type="GP",
            acquisition_type="EI",
            exact_feval=False,
            initial_design_numdata= n_init
        )


    def maximize(self, n_iter=1):
        self.optimizer.run_optimization(max_iter=n_iter, verbosity=False)
        return self.optimizer.Y_best[(self.n_init ):] # exclude the initial samples

    # 1D & 2D Examples
    def example_1d(self):
        # 1D Example
        def f(x):
            return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

        bounds = [{"name": "var_1", "type": "continuous", "domain": (0, 1)}]
        my_problem = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
        my_problem.run_optimization(max_iter=15)
        return my_problem.fx_opt

    def example_2d(self):
        # 2D Example
        def f(x):
            return (
                (x[:, 0] - 3.3) ** 2
                + (x[:, 1] - 1.7) ** 2
                + np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1])
            )

        bounds = [
            {"name": "var_1", "type": "continuous", "domain": (0, 5)},
            {"name": "var_2", "type": "continuous", "domain": (0, 5)},
        ]
        my_problem = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
        my_problem.run_optimization(max_iter=50)
        return my_problem.fx_opt

    # 5D Usage
    def example_5d(self):
        # 5D Example
        def f(x):
            return (
                (x[:, 0] - 3.3) ** 2
                + (x[:, 1] - 1.7) ** 2
                + (x[:, 2] - 1.5) ** 2
                + (x[:, 3] - 2.5) ** 2
                + (x[:, 4] - 3.5) ** 2
                + np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1])
            )

        bounds = [
            {
                "name": f"var_{i+1}",
                "type": "continuous",
                "domain": (-2.0 * np.pi, 2.0 * np.pi),
            }
            for i in range(5)
        ]
        my_problem = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
        my_problem.run_optimization(max_iter=100)
        return my_problem.fx_opt

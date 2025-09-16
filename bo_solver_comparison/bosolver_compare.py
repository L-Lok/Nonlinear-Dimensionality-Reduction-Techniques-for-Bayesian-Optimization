import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from objective_functions.wrappers.gpyopt_wrapper import GPyOptWrapper
from objective_functions.wrappers.botorch_wrapper import BoTorchWrapper
from objective_functions.wrappers.bayes_opt_wrapper import BayesOptWrapper

from objective_functions.objective_functions.rosenbrock import rosenbrock_function
from objective_functions.objective_functions.beale import beale_function
from objective_functions.objective_functions.Hartmann6 import hartman6
from objective_functions.objective_functions.Hartmann3 import hartman3
from objective_functions.objective_functions.Shekel5 import shekel5

from objective_functions.objective_functions.rastrigin import rastrigin_function

# Generalisation to include the use of CUDA
def wrap_function(func):
    return lambda x: func(np.array(x))  # Ensure dim matched

# Target functions
target_functions = [
    beale_function,
    beale_function,
    rosenbrock_function,
    rosenbrock_function,
    hartman3,
    hartman3,
    hartman6,
    hartman6,
    shekel5,
    shekel5,
    rastrigin_function,
    rastrigin_function
]
bounds = [
    [(-4.5, 4.5)] * 2,
    [(-4.5, 4.5)] * 2,
    [(-5, 10)] * 3,
    [(-5, 10)] * 3,
    [(0, 1)] * 3,
    [(0, 1)] * 3,
    [(0, 1)] * 6,
    [(0, 1)] * 6,
    [(0, 10)] * 4,
    [(0, 10)] * 4,
    [(-5.12, 5.12)]*5,
    [(-5.12, 5.12)]*5
]
dims = [2, 2, 3, 3, 3, 3, 6, 6, 4, 4, 5, 5]

# set the baseline for iterations; Simplex Gradients
alpha = 50
iters = [alpha * (d + 1) for d in dims]

##################
# for test only
# iters = [5] * 10
##################

# Collect all results
gpyopt_results = {}
botorch_results = {}
bayes_opt_results = {}

# Run tests
for i, target_function in enumerate(target_functions):
    wrapped_function = wrap_function(target_function)

    # Create Wrappers
    print(f"@ {i + 1}th Start optimisation...")

    gpyopt_wrapper = GPyOptWrapper(wrapped_function, bounds[i], n_init=2 * dims[i])

    botorch_wrapper = BoTorchWrapper(wrapped_function, bounds[i], n_init=2 * dims[i])

    bayes_opt_wrapper = BayesOptWrapper(
        wrapped_function, bounds[i], allow_duplicate_points=True
    )

    # problem number = [f'{i + 1}']
    print("solver: GpyOpt...")
    gpyopt_results[f"{i + 1}"] = gpyopt_wrapper.maximize(n_iter=iters[i]).tolist()
    print("solver: Botorch...")
    botorch_results[f"{i + 1}"] = (
        botorch_wrapper.maximize(n_iter=iters[i]).cpu().detach().numpy().tolist()
    )
    print("solver: Bayes_Opt...")
    bayes_opt_results[f"{i + 1}"] = bayes_opt_wrapper.maximize(n_iter=iters[i]).tolist()


# Define the file path where you want to save the JSON
print("Saving results...")

# Convert dictionaries to DataFrames and save as CSV
for solver_name, results in zip(
    ["gpyopt", "botorch", "bayesopt"],
    [gpyopt_results, botorch_results, bayes_opt_results],
):

    # Flatten the results dictionary into a list of rows for each problem
    rows = []
    for problem_id, result_list in results.items():
        for iteration, result in enumerate(result_list, start=1):
            # Ensure the result is always a list (concatenation-safe)
            if isinstance(result, list):
                rows.append([problem_id, iteration] + result)
            else:
                rows.append([problem_id, iteration, result])  # For single value results

    # Convert to DataFrame
    df = pd.DataFrame(
        rows, columns=["Problem", "Iteration", "Objective"]
    )  # Adjust the column names based on the dimension of your input

    # Save to CSV
    csv_file_path = f"{solver_name}_results.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"Saved results to {csv_file_path}")

import numpy as np
import pandas as pd
import json
from rembo_alg import rembo
from oneHD_low_rank_functions.Ackley import ackley_HD
from oneHD_low_rank_functions.Rosenbrock import Rosenbrock_HD
from oneHD_low_rank_functions.Shekel5 import shekel5_HD
from oneHD_low_rank_functions.Shekel7 import shekel7_HD
from oneHD_low_rank_functions.Styblinski_Tang import StyTang_HD

# 10 Target functions
target_functions = [
    ackley_HD,
    ackley_HD,
    Rosenbrock_HD,
    Rosenbrock_HD,
    shekel5_HD,
    shekel5_HD,
    shekel7_HD,
    shekel7_HD,
    StyTang_HD,
    StyTang_HD,
]

# ambient space dim
d_orig = 100

# effective dim
d_e = 4
##############################
# Note:
# the embedding space dim d = d_e + 1
d = d_e + 1
##############################
# all 600 iterations
##############################
# set the baseline for iterations; Simplex Gradients
alpha = 100
iters = [alpha * (d + 1)] * 10

##################
# for test only
# iters = [5] * 10
##################
rembo_results = {}
rembo_x0 = {}

for i, target_function in enumerate(target_functions):
    print(f"@ {i + 1}th Start optimisation...")

    f0, best_f = rembo(
        d_e=d_e,
        d_orig=d_orig,
        n_init=2 * d,
        n_iter=iters[i],
        high_dim_low_rank_obj_f=target_function,
    )

    rembo_results[f"{i + 1}"] = best_f.tolist()
    rembo_x0[f"{i + 1}"] = f0.tolist()

# Define the file path where you want to save the JSON
print("Saving results...")

# Convert dictionaries to DataFrames and save as CSV
for solver_name, results in zip(["rembo"], [rembo_results]):

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

print("saving f0...")

f0_file_path = "f0_results.json"

# Save the dictionary to a JSON file
with open(f0_file_path, "w") as json_file:
    json.dump(rembo_x0, json_file)

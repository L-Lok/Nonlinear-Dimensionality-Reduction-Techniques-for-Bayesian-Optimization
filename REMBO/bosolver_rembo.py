import json
import pandas as pd
import torch, gpytorch
from oneHD_low_rank_functions.Ackley import ackley_HD
from oneHD_low_rank_functions.Rosenbrock import Rosenbrock_HD
from oneHD_low_rank_functions.Shekel5 import shekel5_HD
from oneHD_low_rank_functions.Shekel7 import shekel7_HD
from oneHD_low_rank_functions.Styblinski_Tang import StyTang_HD


from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# BoTorch Wrapper
class BoTorchWrapper:
    def __init__(self, func, bounds, n_init):
        self.func = func
        self.n_init = n_init
        self.bounds = torch.tensor(bounds, dtype=torch.float64).T.to(
            device
        )  # Transpose to reshape 2 x d

    def gp_model_unit_cube(
        self, train_x: torch.Tensor, train_y: torch.Tensor, covar_module=None
    ):

        # define the likelihood
        g_likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # define the model
        gp_model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=torch.full_like(train_y, 1e-6),
            likelihood=g_likelihood,
            covar_module=covar_module,
            # input/output transformation ref: https://github.com/pytorch/botorch/issues/1150#issuecomment-1086394146
            # input transformation to unit cube
            input_transform=Normalize(d=train_x.shape[1]),
            # one dimensional functional output
            outcome_transform=Standardize(m=1),
        ).to(train_x)
        return gp_model

    def train_gp_plain(self, gp_model: SingleTaskGP) -> SingleTaskGP:
        """
        Train a GP model using the built-in fit_gpytorch_mll.

        Args:
            train_x (torch.Tensor): training inputs with shape (n, d)
            train_y (torch.Tensor): training targets with shape (n, 1)

        Returns:
            SingleTaskGP: trained GP model
        """

        # define the marginal log likelihood
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(device)

        # train the model
        gp_model.train()
        fit_gpytorch_mll(mll)

        # set the model to eval mode
        gp_model.eval()
        # return gp_model

    def _init_set_up(self):
        self.train_x = (
            torch.rand(self.n_init, self.bounds.shape[1], dtype=torch.float64).to(
                device
            )
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )
        self.train_y = self.func(self.train_x.T)  
        if self.train_y.dim != 2:
            self.train_y = self.train_y.unsqueeze(-1)

    def maximize(self, n_iter=1):
        self._init_set_up()
        best_y_init = torch.max(self.train_y.clone())
        best_y = torch.empty(0).to(device)
        for _ in range(n_iter):
            model = self.gp_model_unit_cube(train_x=self.train_x, train_y=self.train_y)
            self.train_gp_plain(model)
            EI = ExpectedImprovement(model, best_f=self.train_y.max())
            candidate, _ = optimize_acqf(
                acq_function=EI,
                bounds=self.bounds,
                q=1,
                num_restarts=256,
                raw_samples=512,
                return_best_only=True,
                timeout_sec=5
            )
            new_y = self.func(candidate.T) 
            if new_y != 2:
                new_y = new_y.unsqueeze(-1)
            self.train_x = torch.cat((self.train_x, candidate))
            self.train_y = torch.cat((self.train_y, new_y))
            best_y = torch.concat((best_y, torch.max(self.train_y).unsqueeze(-1)))

        return best_y_init*(-1), best_y * (-1)


# Target functions
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
bounds = [[(-1, 1)] * 100] * 10
dims = [100] * 10

# set the baseline for iterations; Simplex Gradients
alpha = 6 
iters = [alpha * (d) for d in dims]

##################
# for test only
# iters = [10] * 10
##################

# Collect all results

botorch_results = {}

botorch_f0 = {}
# Run tests
for i, target_function in enumerate(target_functions):

    # Create Wrappers
    print(f"@ {i + 1}th Start optimisation...")

    botorch_wrapper = BoTorchWrapper(target_function, bounds[i], n_init=2 * dims[i])

    # problem number = [f'{i + 1}']
    print("solver: Botorch...")
    best_y_init, best_y = botorch_wrapper.maximize(n_iter=iters[i])
    botorch_f0[f"{i+1}"] = best_y_init.cpu().detach().numpy().tolist()
    botorch_results[f"{i + 1}"] = best_y.cpu().detach().numpy().tolist()

# Define the file path where you want to save the JSON
print("Saving results...")

# Convert dictionaries to DataFrames and save as CSV
for solver_name, results in zip(["bo_lowrank"], [botorch_results]):

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

botorch_file_path = "botorch_f0.json"

# # Save the dictionary to a JSON file
with open(botorch_file_path, "w") as json_file:
    json.dump(botorch_f0, json_file)

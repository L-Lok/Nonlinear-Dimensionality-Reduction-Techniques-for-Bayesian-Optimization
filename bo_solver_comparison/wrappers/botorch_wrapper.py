import torch, gpytorch
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
            train_Yvar=torch.full_like(train_y, 1e-2),
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
        self.train_y = -self._evaluate(self.train_x)  # for minimisation 
        # self.best_y = torch.max(self.train_y.clone()).values
        # self.model = self.gp_model_unit_cube(train_x=self.train_x, train_y=self.train_y)
        # self.train_gp_plain(self.model)

    def _evaluate(self, x):
        return (
            torch.tensor([self.func(xi.cpu().numpy()) for xi in x], dtype=torch.float64)
            .to(device)
            .unsqueeze(-1)
        )

    def maximize(self, n_iter=1):
        self._init_set_up()
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
                timeout_sec=2
            )
            new_y = -self._evaluate(candidate)  # for minimisation 
            self.train_x = torch.cat((self.train_x, candidate))
            self.train_y = torch.cat((self.train_y, new_y))
            best_y = torch.concat((best_y, torch.max(self.train_y).unsqueeze(-1)))

        return best_y * (-1)

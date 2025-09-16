import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data_generator import DatasetPrepare
from .model import LSBOVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAEModel_Plot:
    def __init__(self, hparams, model_path, result_path, train_path, test_path):
        self.train_loader, self.test_loader = DatasetPrepare(hparams=hparams).load(
            train_path=Path(train_path), test_path=Path(test_path)
        )
        self.model = LSBOVAE(hparams=hparams).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.hparams = hparams
        # Load the JSON file
        with open(f"{result_path}", "r") as file:
            self.result = json.load(file)

    def vae_plot(self, output_path):
        """
        For latent dimension >= 3

        Plot the histograms
        """
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))

        latent_test_data = []
        recon_test_data = []
        for test_x, _ in self.test_loader:

            mu, logvar = self.model.encoder(test_x.to(device))
            # print(test_x, mu, logvar)

            # z = torch.distributions.Normal(
            #     loc=mu.mean(axis=0), scale=torch.exp(logvar / 2).mean(axis=0)
            # ).rsample(sample_shape=torch.Size([test_x.shape[0]]))
            z = self.model.sample_latent(mu=mu, logvar=logvar)
            # print(z, self.model.decoder(z));exit()

            # recon_x = self.model.decoder(z)
            recon_x = torch.distributions.Normal(
                self.model.decoder(z), torch.exp(0.5 * self.model.decoder(z))
            ).rsample()
            # print(recon_x[0])
            # print((test_x[0]))
            # exit()
            # Append data
            latent_test_data.append(
                np.ravel(z.clone().to("cpu").detach().numpy()))
            recon_test_data.append(
                np.ravel(recon_x.clone().to("cpu").detach().numpy()))

        recon_train_data = []
        for train_x, _ in self.train_loader:
            mu, logvar = self.model.encoder(train_x.to(device))
            z = self.model.sample_latent(mu=mu, logvar=logvar)
            recon_x = self.model.decoder(z)

            # Append data
            recon_train_data.append(
                np.ravel(recon_x.clone().to("cpu").detach().numpy())
            )

        flatten_test_data = np.ravel(
            self.test_loader.dataset.tensors[0].to("cpu").detach().numpy()
        )
        flatten_train_data = np.ravel(
            self.train_loader.dataset.tensors[0].to("cpu").detach().numpy()
        )

        # ax1 - latent data histogram
        # prior_mu, prior_sigma = 0, 1
        # priors = (
        #     lambda x: 1
        #     / (prior_sigma * np.sqrt(2 * np.pi))
        #     * np.exp(-((x - prior_mu) ** 2) / (2 * prior_sigma**2))
        # )
        priors_samples = np.random.multivariate_normal(
            mean=(np.zeros(self.hparams["latent_dim"])),
            cov=np.diag(np.ones(self.hparams["latent_dim"])),
            size=(500),
        )
        _, bins, _ = axes[0, 0].hist(
            np.ravel(latent_test_data), bins=100, density=True, alpha=0.5
        )
        axes[0, 0].hist(
            np.ravel(priors_samples),
            bins=100,
            label="Standard Normal",
            zorder=-1,
            density=True,
        )
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("Bins of Value")
        axes[0, 0].set_ylabel("Number of Points in Each Bin")
        axes[0, 0].set_title("Latent Data Distribution")

        # ax2 - visualisation of the training loss
        axes[0, 1].plot(np.arange(len(self.result["train"])),
                        self.result["train"])
        axes[0, 1].set_xlabel("iteration")
        axes[0, 1].set_ylabel("training loss")
        axes[0, 1].set_title("Trainig Loss")

        # ax3 - Recon. Test Data Distribution
        axes[1, 0].hist(
            np.ravel(recon_test_data), bins=100, density=True, label="Recon.", alpha=0.5
        )
        axes[1, 0].hist(
            flatten_test_data, bins=100, density=True, label="Orig.", zorder=-1
        )
        axes[1, 0].legend()
        axes[1, 0].set_xlabel("Bins of Values")
        axes[1, 0].set_ylabel("Numbers of Points in Each Bins")
        axes[1, 0].set_title("Test set Reconstruction")

        # ax4 - Recon. Train Data Distribution
        axes[1, 1].hist(
            np.ravel(recon_train_data),
            bins=100,
            density=True,
            label="Recon.",
            alpha=0.5,
        )
        axes[1, 1].hist(
            flatten_train_data, bins=100, density=True, label="Orig.", zorder=-1
        )
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Bins of Values")
        axes[1, 1].set_ylabel("Numbers of Points in Each Bins")
        axes[1, 1].set_title("Train set Reconstruction")

        fig.savefig(output_path / "vae_plot.png")

    def vae_plot_2d(self, output_path):
        """
        For latent dimension = 2

        Plot the histograms
        """
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))

        latent_test_data_comp_1 = []
        latent_test_data_comp_2 = []
        recon_test_data = []

        for test_x, _ in self.test_loader:
            mu, logvar = self.model.encoder(test_x.to(device))
            z = self.model.sample_latent(mu=mu, logvar=logvar)
            recon_x = self.model.decoder(z)

            # print( z.clone().to("cpu").detach().numpy());exit()
            # Append data
            latent_test_data_comp_1 = np.concatenate(
                (latent_test_data_comp_1, z.clone()
                 [:, 0].to("cpu").detach().numpy())
            )
            latent_test_data_comp_2 = np.concatenate(
                (latent_test_data_comp_2, z.clone()
                 [:, 1].to("cpu").detach().numpy())
            )
            recon_test_data.append(
                np.ravel(recon_x.clone().to("cpu").detach().numpy()))

        recon_train_data = []
        for train_x, _ in self.train_loader:
            mu, logvar = self.model.encoder(train_x.to(device))
            z = self.model.sample_latent(mu=mu, logvar=logvar)
            recon_x = self.model.decoder(z)

            # Append data
            recon_train_data.append(
                np.ravel(recon_x.clone().to("cpu").detach().numpy())
            )

        flatten_test_data = np.ravel(
            self.test_loader.dataset.tensors[0].to("cpu").detach().numpy()
        )
        flatten_train_data = np.ravel(
            self.train_loader.dataset.tensors[0].to("cpu").detach().numpy()
        )

        # ax1 - latent data histogram
        priors = np.random.multivariate_normal(
            mean=[0, 0], cov=np.diag([1.0, 1.0]), size=len(latent_test_data_comp_1)
        )
        axes[0, 0].scatter(
            latent_test_data_comp_1, latent_test_data_comp_2, label="laten data"
        )
        axes[0, 0].scatter(
            priors[:, 0], priors[:, 1], label="Standard Normal", zorder=-1
        )
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("z1")
        axes[0, 0].set_ylabel("z2")
        axes[0, 0].set_title("Latent Data Distribution")

        # ax2 - visualisation of the training loss
        axes[0, 1].plot(np.arange(len(self.result["train"])),
                        self.result["train"])
        axes[0, 1].set_xlabel("iteration")
        axes[0, 1].set_ylabel("training loss")
        axes[0, 1].set_title("Trainig Loss")

        # ax3 - Recon. Test Data Distribution
        axes[1, 0].hist(
            np.ravel(recon_test_data), bins=100, density=True, label="Recon.", alpha=0.5
        )
        axes[1, 0].hist(
            flatten_test_data, bins=100, density=True, label="Orig.", zorder=-1
        )
        axes[1, 0].legend()
        axes[1, 0].set_xlabel("Bins of Values")
        axes[1, 0].set_ylabel("Numbers of Points in Each Bins")
        axes[1, 0].set_title("Test set Reconstruction")

        # ax4 - Recon. Train Data Distribution
        axes[1, 1].hist(
            np.ravel(recon_train_data),
            bins=100,
            density=True,
            label="Recon.",
            alpha=0.5,
        )
        axes[1, 1].hist(
            flatten_train_data, bins=100, density=True, label="Orig.", zorder=-1
        )
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Bins of Values")
        axes[1, 1].set_ylabel("Numbers of Points in Each Bins")
        axes[1, 1].set_title("Train set Reconstruction")

        fig.savefig(output_path / "vae_plot.png")

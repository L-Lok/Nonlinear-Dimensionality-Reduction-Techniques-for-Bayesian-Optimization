import functools
import logging

import numpy as np
import torch
from botorch.utils.probability import TruncatedMultivariateNormal
from torch.utils.data import TensorDataset, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataWeighter:
    weight_types = ["uniform"]
    # Different weights for different data points -- future work

    def __init__(self, hparams):
        if hparams['weight_type'] is not None:
            self.weight_type = hparams['weight_type']
        if hparams['weight_type'] == "uniform":
            self.weighting_function = DataWeighter.uniform_weights
        else:
            raise NotImplementedError
        
    @staticmethod
    def uniform_weights(properties: np.array):
        return torch.ones_like(properties)


class DatasetPrepare(DataWeighter):
    """
    For training pretrained VAE. we do not require metric loss, i.e. no need for function values.
    """

    def __init__(self, hparams: dict):
        super().__init__(hparams)

        self.num_sample = hparams['num_sample']
        self.x_dim = hparams['feature_dim']
        self.target_dim = hparams['target_dim']
        self.data_split = hparams['data_split']
        self.sampling_method = hparams['sampling_method']
        self.batch_size = hparams['batch_size']

    @staticmethod
    def generate_high_correlation_cov_matrix(size, correlation_value=0.9):
        # Ensure the correlation value is between -1 and 1
        assert -1 < correlation_value < 1, "Correlation value must be between -1 and 1"

        # Create a matrix with the desired correlation value
        base_matrix = torch.full((size, size), correlation_value)

        # Set the diagonal elements to 1 (variance of 1 for each variable)
        base_matrix.fill_diagonal_(1)

        # Make the matrix positive definite
        # Add a small value to the diagonal to ensure positive definiteness
        epsilon = 1e-4
        positive_definite_matrix = base_matrix + \
            epsilon * torch.eye(size)

        return positive_definite_matrix

    def data_generator(self, seed=42):
        """
        Args:
            seed (Any, optional): Pytorch seed. Defaults to 42.
        """
        # Calculate the sizes for the train and test splits
        train_size = int(self.data_split * self.num_sample)
        test_size = self.num_sample - train_size  # Ensure the sum equals num_sample

        # set the seed
        torch.manual_seed(seed)

        # for function values, it's a high dimensional input with a single output
        # as we do not care about the function values, we can set them to 1
        f_train = torch.ones(train_size, 1)
        f_test = torch.ones(test_size, 1)

        if self.sampling_method == 'uniform':
            # data needs to be highly correlated
            raise NotImplementedError

        elif self.sampling_method == 'trunc_normal':
            # Generate a 5x5 positive definite covariance matrix
            # we start from basic standard normal
            sample_loc = torch.zeros(self.x_dim)
            sample_cov = self.generate_high_correlation_cov_matrix(self.x_dim)

            trunc_normal = TruncatedMultivariateNormal(
                loc=sample_loc,
                covariance_matrix=sample_cov,
                bounds=torch.tensor([[-5., 5.] for _ in range(self.x_dim)])
            )
            x_train = trunc_normal.sample(torch.Size([train_size]))

            x_test = trunc_normal.sample(torch.Size([test_size]))
        elif self.sampling_method == 'normal':
            logging.warning(
                "Normal sampling method is not recommended for VAE training.")
            logging.warning(
                "b/c the data points may exist in the uncorrelated space.")
            # Generate a 5x5 positive definite covariance matrix
            # we start from basic standard normal
            sample_loc = torch.zeros(self.x_dim)
            sample_cov = self.generate_high_correlation_cov_matrix(self.x_dim)

            x_train = torch.distributions.MultivariateNormal(
                loc=sample_loc, covariance_matrix=sample_cov
            ).sample(torch.Size([train_size]))

            x_test = torch.distributions.MultivariateNormal(
                loc=sample_loc, covariance_matrix=sample_cov
            ).sample(torch.Size([test_size]))
        else:
            raise ValueError(f'Unsupported {self.sampling_method} !')

        print("x_train", x_train.shape)
        print("f_train", f_train.shape)
        print("x_test", x_test.shape)
        print("f_test", f_test.shape)
        return x_train.to(device), f_train.to(device), x_test.to(device), f_test.to(device)

    def setup(self):
        print("dataset setup")

        x_train, f_train, x_test, f_test = self.data_generator()

        self.train_weights = self.weighting_function(f_train).flatten()
        self.test_weights = self.weighting_function(f_test).flatten()

        # Create Sampler
        # Different weights for different data points -- future work
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )
        self.test_sampler = WeightedRandomSampler(
            self.test_weights, num_samples=len(self.test_weights), replacement=True
        )
        ######################################################################
        self.train_dataset = TensorDataset(x_train, f_train)
        self.test_dataset = TensorDataset(x_test, f_test)

        print(self.train_dataset.tensors[0].shape)
        print(self.train_dataset.tensors[1].shape)
        print(self.test_dataset.tensors[0].shape)
        print(self.test_dataset.tensors[1].shape)
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            sampler=self.train_sampler,
            drop_last=True
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            sampler=self.test_sampler,
            drop_last=True
        )

        return train_dataloader, test_dataloader

    def load(self, train_path, test_path,drop_last=True):
        print("dataset load")

        x_train, f_train = torch.load(train_path)
        x_test, f_test = torch.load(test_path)
        return self.load_tensor(x_train, f_train, x_test, f_test,drop_last=drop_last)

    def load_tensor(self, x_train, f_train, x_test, f_test,drop_last=True):
        self.train_weights = self.weighting_function(f_train).flatten()
        self.test_weights = self.weighting_function(f_test).flatten()

        # Create Sampler
        # Different weights for different data points -- future work
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )
        self.test_sampler = WeightedRandomSampler(
            self.test_weights, num_samples=len(self.test_weights), replacement=True
        )
        #################################################################
        self.train_dataset = TensorDataset(x_train, f_train)
        self.test_dataset = TensorDataset(x_test, f_test)

        print(self.train_dataset.tensors[0].shape)
        print(self.train_dataset.tensors[1].shape)
        print(self.test_dataset.tensors[0].shape)
        print(self.test_dataset.tensors[1].shape)

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            sampler=self.train_sampler,
            drop_last=drop_last,
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            sampler=self.test_sampler,
            drop_last=drop_last,
        )

        return train_dataloader, test_dataloader

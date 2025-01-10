#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.bayesopt.kernel_function.kernel_list import *
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.containers import TrainingData
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors import GammaPrior, UniformPrior, HorseshoePrior
from torch import Tensor
from gpytorch.distributions.multivariate_normal import MultivariateNormal


class NewMixedSingleTaskGP(SingleTaskGP):
    r"""A single-task exact GP model for categorical search spaces.
    """
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: List[int],
        likelihood: Optional[Likelihood] = None,
        BBM = None
    ) -> None:
        r"""A single-task exact GP model supporting categorical parameters.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            cat_dims: A list of indices corresponding to the columns of
                the input `X` that should be considered categorical features.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
        """
        if len(cat_dims) == 0:
            raise ValueError(
                "Must specify categorical dimensions for MixedSingleTaskGP"
            )
        self._ignore_X_dims_scaling_check = cat_dims
        input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        if likelihood is None:
            # This Gamma prior is quite close to the Horseshoe prior
            min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
            likelihood = GaussianLikelihood(       
                batch_shape=aug_batch_shape,
                noise_constraint=GreaterThan(
                    min_noise, transform=None, initial_value=1e-3
                ),
                noise_prior=GammaPrior(0.9, 10.0),      # noise variance
            )

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)

        lengthscale_prior = GammaPrior(3.0,6.0) # \frac{1}{\beta_i}
        outputscale_prior = UniformPrior(0,1,validate_args=False)
        outputscale_constraint = Interval(0,1, initial_value=0.1)
        outputscale_prior.low = outputscale_prior.low.cuda()
        outputscale_prior.high = outputscale_prior.high.cuda()
        
        hamming_kernel = ScaleKernel(
            CategoricalKernel2(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                lengthscale_constraint=GreaterThan(1e-06),
                lengthscale_prior=lengthscale_prior
            ),
            batch_shape=aug_batch_shape,
            outputscale_constraint=outputscale_constraint,
            outputscale_prior=outputscale_prior,
        ) 
        
        linear_kernel2 = LinearKernel2(BBM)
        # linear_kernel = LinearKernel()
        # matern_kernel = MaternKernel()
        # periodic_kernel = PeriodicKernel()
        # rbf_kernel = RBFKernel()
        # rqk_kernel = RQKernel()
        
        # self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        # covar_module = linear_kernel + hamming_kernel
        # covar_module = hamming_kernel
        # covar_module = hamming_kernel
        # covar_module = MultiKernel([hamming_kernel, linear_kernel])
        # covar_module = MultiKernel([hamming_kernel, linear_kernel])
        # covar_module = MultiKernel([linear_kernel, hamming_kernel])
        
        # covar_module = PiecewisePolynomialKernel()   # RuntimeError: Expected condition, x and y to be on the same device, but condition is on cpu and x and y are on cuda:0 and cuda:0 respectively
        # covar_module = RBFKernel()
        # covar_module = SpectralDeltaKernel()    # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        # covar_module = SpectralMixtureKernel()  # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        # covar_module = ArcKernel()  # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        # covar_module = IndexKernel()    # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        # covar_module = MultitaskKernel()    # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        # covar_module = GridKernel() # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        # covar_module = GridInterpolationKernel()    # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        # covar_module = RFFKernel()  # TypeError: __init__() missing 1 required positional argument: 'num_dims'
        
        # covar_module = MultiKernel([hamming_kernel, linear_kernel, matern_kernel, periodic_kernel, rbf_kernel, rqk_kernel])
        # covar_module = MultiKernel([hamming_kernel, linear_kernel2])
        covar_module = TwoKernel([hamming_kernel, linear_kernel2])
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
        )
    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(
        cls, training_data: TrainingData, **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "cat_dims": kwargs["categorical_features"],
            "likelihood": kwargs.get("likelihood"),
        }

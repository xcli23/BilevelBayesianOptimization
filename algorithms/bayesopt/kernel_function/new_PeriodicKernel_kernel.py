from gpytorch.kernels.periodic_kernel import PeriodicKernel, Optional ,Interval, Prior
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC
import math

class PeriodicKernel2(PeriodicKernel, KernelABC):
    def __init__(
        self,
        BBM,
        embed,
        period_length_prior: Optional[Prior] = None,
        period_length_constraint: Optional[Interval] = None,
        **kwargs,
    ):  
        self.BBM = BBM
        self.embed = embed
        self.base_kernel = None
        super().__init__(period_length_prior, period_length_constraint, **kwargs)
    
    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = self.convert_to_embedding(x1, x2)
        # Pop this argument so that we can manually sum over dimensions
        last_dim_is_batch = params.pop("last_dim_is_batch", False)
        # Get lengthscale
        lengthscale = self.lengthscale

        x1_ = x1.div(self.period_length / math.pi)
        x2_ = x2.div(self.period_length / math.pi)
        # We are automatically overriding last_dim_is_batch here so that we can manually sum over dimensions.
        diff = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True, **params)

        if diag:
            lengthscale = lengthscale[..., 0, :, None]
        else:
            lengthscale = lengthscale[..., 0, :, None, None]
        exp_term = diff.sin().pow(2.0).div(lengthscale).mul(-2.0)

        if not last_dim_is_batch:
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

        return exp_term.exp()

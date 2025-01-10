import torch
from gpytorch.kernels.linear_kernel import LinearKernel, Optional, Prior, Interval, MatmulLazyTensor, RootLazyTensor
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC

class LinearKernel2(LinearKernel, KernelABC):
    def __init__(
        self,
        BBM,
        embed,
        num_dimensions: Optional[int] = None,
        offset_prior: Optional[Prior] = None,
        variance_prior: Optional[Prior] = None,
        variance_constraint: Optional[Interval] = None,
        **kwargs,
    ):  
        self.BBM = BBM
        self.embed = embed
        super().__init__(num_dimensions, offset_prior, variance_prior, variance_constraint, **kwargs)
        self.base_kernel = None
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1, x2 = self.convert_to_embedding(x1, x2)
        
        x1_ = x1 * self.variance.sqrt()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyTensor(x1_)

        else:
            x2_ = x2 * self.variance.sqrt()
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

            prod = MatmulLazyTensor(x1_, x2_.transpose(-2, -1))

        if diag:
            return prod.diag()
        else:
            return prod
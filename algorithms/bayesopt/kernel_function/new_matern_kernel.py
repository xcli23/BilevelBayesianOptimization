import math
import torch
from gpytorch.kernels.matern_kernel import MaternKernel, Optional, MaternCovariance, trace_mode
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC

class MaternKernel2(MaternKernel, KernelABC):
    def __init__(
        self,
        BBM,
        embed,
        nu: Optional[float] = 2.5, 
        **kwargs
    ):  
        self.BBM = BBM
        self.embed = embed
        self.base_kernel = None
        super().__init__(nu, **kwargs)
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1, x2 = self.convert_to_embedding(x1, x2)
        
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC
import torch
from torch import Tensor


import torch
import torch.nn.functional as F
from gpytorch.kernels.kernel import Kernel
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC

class HIKernel2(Kernel, KernelABC):
    
    has_lengthscale = True
    
    def __init__(self, batch_shape, ard_num_dims,lengthscale_constraint,lengthscale_prior, BBM):
        self.BBM = BBM
        super(HIKernel2, self).__init__(batch_shape=batch_shape, ard_num_dims=ard_num_dims, lengthscale_constraint=lengthscale_constraint, lengthscale_prior=lengthscale_prior)

    has_lengthscale = False 


    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        key: int = 0,
        **kwargs
    ) -> Tensor:
        """
        Forward propagation method to compute the histogram intersection kernel between two input tensors.

        Args:
            x1 (Tensor): The first input tensor with shape [..., C], where C is the number of categories.
            x2 (Tensor): The second input tensor with shape [..., C], where C is the number of categories.
            diag (bool, optional): Whether to return only the diagonal elements. Default is False.
            last_dim_is_batch (bool, optional): Whether the last dimension represents the batch dimension. Default is False.
            key (int, optional): Selects different implementation methods. Default is 0.
            **kwargs: Additional unused keyword arguments.

        Returns:
            Tensor: The computed kernel matrix, with a shape determined by the inputs and parameters.
        """

        if key == 0:
            min_vals = torch.min(x1.unsqueeze(-2), x2.unsqueeze(-3))
            sum_min = min_vals.sum(-1)
        elif key == 1:
            min_vals = torch.min(x1.unsqueeze(-2), x2.unsqueeze(-3))
            sum_min = min_vals.sum(-1)
            if last_dim_is_batch:
                sum_min = sum_min.transpose(-3, -1)
            else:
                sum_min = sum_min.mean(-1)
        else:
            raise ValueError("Invalid key value for HIKernel2")
        sum_x1 = x1.sum(-1).unsqueeze(-1)
        sum_x2 = x2.sum(-1).unsqueeze(-2)
        norm_factor = torch.sqrt(sum_x1 * sum_x2)

        norm_factor = norm_factor + 1e-8

        res = sum_min / norm_factor 

        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)

        return res
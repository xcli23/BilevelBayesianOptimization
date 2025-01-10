import torch
import torch.nn.functional as F
from gpytorch.kernels.kernel import Kernel
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC

class XingKernel(Kernel, KernelABC):
    
    has_lengthscale = True
    
    def __init__(self, batch_shape, ard_num_dims,lengthscale_constraint,lengthscale_prior, BBM, embed):
        self.BBM = BBM
        self.embed = embed
        super(XingKernel, self).__init__(batch_shape=batch_shape, ard_num_dims=ard_num_dims, lengthscale_constraint=lengthscale_constraint, lengthscale_prior=lengthscale_prior)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1, x2 = self.convert_to_embedding(x1, x2)
        
        ret_1_normalized = F.normalize(x1, p=2, dim=1)
        ret_2_normalized = F.normalize(x2, p=2, dim=1)
        result = torch.matmul(ret_1_normalized, ret_2_normalized.t())

        if diag:
            result = torch.diagonal(result, dim1=-1, dim2=-2)
        
        return result

class XingKernel2(Kernel, KernelABC):
    
    has_lengthscale = True
    
    def __init__(self, BBM, embed, batch_shape=torch.Size([]), ard_num_dims=None,lengthscale_constraint=None,lengthscale_prior=None):
        self.BBM = BBM
        self.embed = embed
        super(XingKernel2, self).__init__(batch_shape=batch_shape, ard_num_dims=ard_num_dims, lengthscale_constraint=lengthscale_constraint, lengthscale_prior=lengthscale_prior)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1, x2 = self.convert_to_embedding(x1, x2)
        
        ret_1_normalized = F.normalize(x1, p=2, dim=1)
        ret_2_normalized = F.normalize(x2, p=2, dim=1)
        result = torch.matmul(ret_1_normalized, ret_2_normalized.t())

        if diag:
            result = torch.diagonal(result, dim1=-1, dim2=-2)
        
        return result
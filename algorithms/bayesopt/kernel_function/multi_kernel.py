import torch
import torch.nn.functional as F
from gpytorch.kernels.kernel import Kernel
from gpytorch.lazy import MatmulLazyTensor
from gpytorch.lazy import SumLazyTensor
from torch import Tensor
import numpy as np
from textattack.shared.utils import LazyLoader
import algorithms.glo as glo


class TwoKernel(Kernel):
    def __init__(self, kernels:list):
        super(TwoKernel, self).__init__()
        self.base_kernel = None
        for kernel in kernels:
            if hasattr(kernel, "base_kernel") and kernel.base_kernel.__class__.__name__ == 'CategoricalKernel2':
                self.base_kernel = kernel.base_kernel  
        self.kernels = torch.nn.ModuleList(kernels)
        self.weight = torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True)  
    
    def forward(self, x1, x2, **params):
        weight = glo.get_value('bo_weight')[0]
        return weight * self.kernels[0](x1, x2, **params) + (1 - weight) * self.kernels[1](x1, x2, **params) 
    
class MultiKernel(Kernel):
    def __init__(self, kernels: list):
        super(MultiKernel, self).__init__()
        self.base_kernel = None
        for kernel in kernels:
            if hasattr(kernel, "base_kernel") and kernel.base_kernel.__class__.__name__ == 'CategoricalKernel2':
                self.base_kernel = kernel.base_kernel  
        self.kernels = torch.nn.ModuleList(kernels)
        self.kernels_num = len(self.kernels)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor([1.0/self.kernels_num]), requires_grad=True) for _ in range(self.kernels_num)])
    
    def forward(self, x1, x2, **params):
        weights = glo.get_value('bo_weight')
        output = weights[0] * self.kernels[0](x1, x2, **params)
        for i in range(1, self.kernels_num):
            output +=  weights[i] * self.kernels[i](x1, x2, **params)
        return output

class WeightsAndBiasesLogger():
    """Logs attack results to Weights & Biases."""

    def __init__(self, **kwargs):

        global wandb
        wandb = LazyLoader("wandb", globals(), "wandb")

        wandb.init(**kwargs)
        self.kwargs = kwargs
        self.project_name = wandb.run.project_name()
        self._result_table_rows = []

    def __setstate__(self, state):
        global wandb
        wandb = LazyLoader("wandb", globals(), "wandb")

        self.__dict__ = state
        wandb.init(resume=True, **self.kwargs)

    def set_logger(self, logger:dict):
        wandb.log(logger)
    
    def finish(self):
        wandb.finish()
    

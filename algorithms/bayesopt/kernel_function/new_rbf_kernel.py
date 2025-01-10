from gpytorch.kernels.rbf_kernel import RBFKernel,RBFCovariance,trace_mode
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC

def postprocess_rbf(dist_mat):
        return dist_mat.div_(-2).exp_()

class RBFKernel2(RBFKernel, KernelABC):
    def __init__(
        self,
        BBM,
        embed,
        **kwargs,
    ):  
        self.BBM = BBM
        self.embed = embed
        super(RBFKernel2, self).__init__(**kwargs)
        self.base_kernel = None
    
    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = self.convert_to_embedding(x1, x2)
        
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
            ),
        )

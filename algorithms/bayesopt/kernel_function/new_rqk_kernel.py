from __future__ import absolute_import, division, print_function, unicode_literals
from gpytorch.kernels.rq_kernel import RQKernel, Optional, Interval
from algorithms.bayesopt.kernel_function.kernel_abc import KernelABC


class RQKernel2(RQKernel, KernelABC):
    def __init__(
        self,
        BBM,
        embed,
        alpha_constraint: Optional[Interval] = None,
        **kwargs,
    ):  
        self.BBM = BBM
        self.embed = embed
        self.base_kernel = None
        super().__init__(alpha_constraint, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = self.convert_to_embedding(x1, x2)
        
        def postprocess_rq(dist):
            alpha = self.alpha
            for _ in range(1, len(dist.shape) - len(self.batch_shape)):
                alpha = alpha.unsqueeze(-1)
            return (1 + dist.div(2 * alpha)).pow(-alpha)

        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rq, postprocess=True, **params
        )
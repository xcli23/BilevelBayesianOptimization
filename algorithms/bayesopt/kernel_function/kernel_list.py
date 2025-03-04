from gpytorch.kernels.cosine_kernel import CosineKernel
from gpytorch.kernels.cylindrical_kernel import CylindricalKernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.periodic_kernel import PeriodicKernel
from gpytorch.kernels.piecewise_polynomial_kernel import PiecewisePolynomialKernel
from gpytorch.kernels.polynomial_kernel import PolynomialKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.rq_kernel import RQKernel
from gpytorch.kernels.spectral_delta_kernel import SpectralDeltaKernel
from gpytorch.kernels.spectral_mixture_kernel import SpectralMixtureKernel
from gpytorch.kernels.arc_kernel import ArcKernel
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.lcm_kernel import MultitaskKernel
from gpytorch.kernels.grid_kernel import GridKernel
from gpytorch.kernels.grid_interpolation_kernel import GridInterpolationKernel
from gpytorch.kernels.rff_kernel import RFFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

from algorithms.bayesopt.kernel_function.new_categorical_kernel import CategoricalKernel2 # Memory efficient re-implementation of the categorical kernel
from algorithms.bayesopt.kernel_function.xing_kernel import XingKernel, XingKernel2
from algorithms.bayesopt.kernel_function.new_linear_kernel import LinearKernel2
from algorithms.bayesopt.kernel_function.new_matern_kernel import MaternKernel2
from algorithms.bayesopt.kernel_function.new_PeriodicKernel_kernel import PeriodicKernel2
from algorithms.bayesopt.kernel_function.new_rbf_kernel import RBFKernel2
from algorithms.bayesopt.kernel_function.new_rqk_kernel import RQKernel2
from algorithms.bayesopt.kernel_function.histogram_intersection_kernel import HIKernel2
from algorithms.bayesopt.kernel_function.multi_kernel import TwoKernel
from algorithms.bayesopt.kernel_function.multi_kernel import MultiKernel
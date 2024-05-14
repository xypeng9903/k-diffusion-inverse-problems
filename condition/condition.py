import warnings
import torch
from torch import nn
from torch.fft import fft2, ifft2
from abc import abstractmethod
from condition.diffpir_utils.utils_sisr import pre_calculate
from linear_operator.utils import linear_cg

import condition.diffpir_utils.utils_sisr as sr
from .measurements import convert_to_linear_op
from k_diffusion.external import OpenAIDenoiserV2
from .covariance import (
    IsotropicCovariance,
    DiagonalCovariance,
    TweedieCovariance,
    DiscreteWaveletTransform,
    LikelihoodCovariance
)


class ConditionDenoiser(nn.Module):
    '''Approximate E[x0|xt, y] given variational Gaussian posterior'''

    def __init__(
        self,
        operator, 
        measurement,
        device='cpu'
    ):
        super().__init__()
        self.operator = operator
        self.measurement = measurement
        self.device = device

    @abstractmethod
    def uncond_pred(self, x, sigma):
        raise NotImplementedError
    
    @abstractmethod
    def covar_schedule(self, tweedie_covar, learned_covar, sigma):
        raise NotImplementedError

    def forward(self, x, sigma):
        assert x.shape[0] == 1
        x0_mean, tweedie_covar, learned_covar = self.uncond_pred(x, sigma)
        covar1, covar2 = self.covar_schedule(tweedie_covar, learned_covar, sigma)
        mat = MatSolver(self.operator, self.measurement, x0_mean, covar2, sigma).solve()
        correction = (covar1 @ mat).reshape(x0_mean.shape)
        hat_x0 = x0_mean + correction
        return hat_x0.clip(-1, 1).detach()
    

class ConditionOpenAIDenoiserV2(ConditionDenoiser):

    def __init__(
        self, 
        denoiser: OpenAIDenoiserV2,
        guidance: str = "I-1",
        **kwargs
    ):
        r"""
        Args: covar_schedule
        """
        super().__init__(**kwargs)
        self.denoiser = denoiser.requires_grad_(False)
        self.guidance = guidance
        self.ortho_tf_type = denoiser.ortho_tf_type

    def uncond_pred(self, x: torch.Tensor, sigma):
        c_out, c_in = self.denoiser.get_scalings(sigma)
        model_output, logvar, logvar_ot = self.denoiser(x, sigma, return_variance=True)
        x0_mean = model_output * c_out + x
        predict_covar = self._get_predict_covar(logvar_ot.exp() * c_out**2)
        tweedie_covar = TweedieCovariance(x, self.denoiser, sigma)
        return x0_mean, tweedie_covar, predict_covar
    
    def covar_schedule(self, tweedie_covar, predict_covar, sigma):
        pgdm_covar = IsotropicCovariance(sigma.pow(2) / (1 + sigma.pow(2)), 3*256*256)
        guidance = self.guidance.split('-')
        if guidance[0] == 'I':
            thres = float(guidance[1])
            covar1 = tweedie_covar
            covar2 = predict_covar if sigma < thres else pgdm_covar
        elif guidance[0] == 'II':
            thres = float(guidance[1])
            covar = predict_covar if sigma < thres else pgdm_covar
            covar1 = covar
            covar2 = covar
        elif guidance[0] == 'tweedie':
            thres1, thres2, thres = float(guidance[1]), float(guidance[2]), float(guidance[3])
            # covar1
            if sigma < thres1:
                covar1 = tweedie_covar
            else:
                covar1 = predict_covar
            # covar2
            if sigma < thres2:
                covar2 = tweedie_covar
            elif sigma < thres:
                covar2 = predict_covar
            else:
                covar2 = pgdm_covar
        else:
            raise NotImplementedError
        return covar1, covar2
    
    def _get_predict_covar(self, variance):
        PsiT = DiscreteWaveletTransform(variance.shape[1:])
        Psi = PsiT.transpose(-2, -1)
        diag = DiagonalCovariance(variance.flatten())
        return Psi @ diag @ PsiT
    
    
#------------------------------
# Implementation of mat solver
#------------------------------
    
class MatSolver:
    
    def __init__(
            self, 
            operator, 
            measurement, 
            x0_mean, 
            x0_covar, 
            sigma
        ):
        y, y_flatten = measurement
        self.saved = [operator, y, y_flatten, x0_mean, x0_covar, sigma]
        self.device = x0_mean.device
        
    def solve(self, mode='auto', **kwargs):
        x0_covar_type = type(self.saved[4])
        if mode == 'auto':
            mode = 'closed_form' if x0_covar_type in [IsotropicCovariance] else 'cg'
        
        if mode == 'closed_form':
            return self._closed_form(**kwargs)
        elif mode == 'cg':
            u = self._cg(**kwargs)
        elif mode == 'adam':
            u = self._adam(**kwargs)
        else:
            raise NotImplementedError
        
        operator = self.saved[0]
        A, AT, sigma_s = convert_to_linear_op(operator)
        mat = AT @ u
        return mat
    
    def _cg(self, **kwargs):
        operator, y, y_flatten, x0_mean, x0_covar, sigma = self.saved
        A, AT, sigma_s = convert_to_linear_op(operator)
        y_flatten, x0_mean = y_flatten.reshape(-1, 1), x0_mean.reshape(-1, 1)
        likelihood_covar = LikelihoodCovariance(y_flatten, operator, x0_covar)
        u = linear_cg(likelihood_covar.matmul, y_flatten - A @ x0_mean, **kwargs)
        return u
    
    @torch.enable_grad()
    def _adam(self, **kwargs):
        operator, y, y_flatten, x0_mean, x0_covar, sigma = self.saved
        A, AT, sigma_s = convert_to_linear_op(operator)
        y_flatten, x0_mean = y_flatten.reshape(-1, 1), x0_mean.reshape(-1, 1)
        likelihood_covar = LikelihoodCovariance(y_flatten, operator, x0_covar)
                
        u = torch.zeros_like(y_flatten).requires_grad_()
        optimizer = torch.optim.Adam([u], lr=1)

        # optimizatin loop
        tol = kwargs.get('tol', 1e-3)
        max_iter = kwargs.get('max_iter', 300)
        iteration = 0
        rhs = y_flatten - A @ x0_mean
        rhs_norm = rhs.norm().detach()
        while True:
            loss = (u * (0.5 * likelihood_covar @ u - rhs)).sum() / rhs_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            relative_error = u.grad.norm()                           
            # print(
            #     f"sigma: {sigma.item():.4f}", 
            #     f"iteration: {iteration}",
            #     f"loss: {loss.item():.4f}",
            #     f"relative_error: {relative_error.item():.6f}",
            # )
            if  relative_error < tol:
                break
            if iteration > max_iter:
                warnings.warn(f"Adam not converge with {iteration} iterations. error: {relative_error.item():.6f} (tol: {tol:.6f})")
                break
            iteration += 1
                    
        return u
    
    def _closed_form(self, x0_var=None):
        operator, y, y_flatten, x0_mean, x0_covar, sigma = self.saved
        sigma_s = operator.sigma_s
        if x0_var is None:
            x0_var = x0_covar.variance
        if operator.name == 'inpainting':
            mask = operator.mask
            mat = (mask * y - mask * x0_mean) / (sigma_s.pow(2) + x0_var)
        elif operator.name == 'gaussian_blur' or operator.name == 'motion_blur':
            FB, FBC, F2B, FBFy = pre_calculate(y, operator.get_kernel(), 1)
            mat = ifft2(FBC / (sigma_s.pow(2) + x0_var * F2B) * fft2(y - ifft2(FB * fft2(x0_mean)))).real
        elif operator.name == 'super_resolution':
            sf = operator.scale_factor
            FB, FBC, F2B, FBFy = pre_calculate(y, operator.get_kernel(), sf)
            invW = torch.mean(sr.splits(F2B, sf), dim=-1, keepdim=False)
            mat = ifft2(FBC * (fft2(y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)) / (sigma_s.pow(2) + x0_var * invW)).repeat(1, 1, sf, sf)).real
        else:
            raise NotImplementedError
        return mat.reshape(-1, 1)



    
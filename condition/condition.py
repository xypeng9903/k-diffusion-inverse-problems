import torch
from torch import nn
from torch.autograd import grad
from torch.fft import fft2, ifft2
from scipy.sparse.linalg import cg, LinearOperator
import numpy as np
from warnings import warn
from abc import abstractmethod
import gpytorch
from gpytorch.distributions import MultivariateNormal

import condition.diffpir_utils.utils_sisr as sr
from .utils import OrthoTransform, LazyOTCovariance
from k_diffusion.external import OpenAIDenoiser, OpenAIDenoiserV2
from guided_diffusion.gaussian_diffusion import GaussianDiffusion, _extract_into_tensor


class LazyLikelihoodCovariance(gpytorch.LinearOperator):
    
    def __init__(self, y_flatten, x0_cov, operator):
        super().__init__(y_flatten, x0_cov=x0_cov, operator=operator)
        self.saved = x0_cov, operator

    def _matmul(self: gpytorch.LinearOperator, rhs: torch.Tensor) -> torch.Tensor:
        x0_cov, operator = self.saved
        sigma_s = operator.sigma_s
        A = lambda x: operator.forward(x,  noiseless=True, flatten=True)[1]
        AT = lambda x: operator.transpose(x, flatten=True)
        rhs = rhs.permute(1, 0)
        rhs = sigma_s**2 * rhs + A(x0_cov(AT(rhs)))
        rhs = rhs.permute(1, 0)
        return rhs
    
    def _size(self) -> torch.Size:
        return torch.Size([self._args[0].numel(), self._args[0].numel()])
    
    def _transpose_nonbatch(self: gpytorch.LinearOperator) -> gpytorch.LinearOperator:
        return self


class ConditionDenoiser(nn.Module):
    '''Approximate E[x0|xt, y] given variational Gaussian posterior'''

    def __init__(
        self,
        operator, 
        measurement, 
        guidance,
        device='cpu',
        zeta=None,
        lambda_=None,
        eta=None,
        num_hutchinson_samples=None,
        mle_sigma_thres=0.2,
        ortho_tf_type=None
    ):
        super().__init__()
        self.operator = operator
        self.y, self.y_flatten = measurement
        
        self.guidance = guidance
        self.zeta = zeta
        self.lambda_ = lambda_
        self.eta = eta
        self.num_hutchinson_samples = num_hutchinson_samples
        self.mle_sigma_thres = mle_sigma_thres
        
        self.device = device
        self.ortho_tf_type = ortho_tf_type
        self.ortho_tf = OrthoTransform(ortho_tf_type)
        self.mat_solver = __MAT_SOLVER__[operator.name]

    @abstractmethod
    def uncond_pred(self, x, sigma):
        raise NotImplementedError

    def loglikelihood(self, x0_mean, x0_var, theta0_var):
        x0_cov = LazyOTCovariance(self.ortho_tf, x0_var if self.ortho_tf_type is None else theta0_var)        
        mean = self.operator.forward(x0_mean, noiseless=True, flatten=True)[1][0]
        cov = LazyLikelihoodCovariance(self.y_flatten, x0_cov=x0_cov, operator=self.operator)
        return MultivariateNormal(mean, cov).log_prob(self.y_flatten[0])

    def forward(self, x, sigma):
        assert x.shape[0] == 1

        if self.guidance == "uncond":
            hat_x0 = self.uncond_pred(x, sigma)[0]

        elif self.guidance == "autoI":
            hat_x0 = self._auto_type_I_guidance_impl(x, sigma)

        elif self.guidance == "I":
            hat_x0 = self._type_I_guidance_impl(x, sigma)

        elif self.guidance == "II":
            hat_x0 = self._type_II_guidance_impl(x, sigma)

        elif self.guidance == "dps":
            hat_x0 = self._dps_guidance_impl(x, sigma)

        elif self.guidance == "pgdm":
            hat_x0 = self._pgdm_guidance_impl(x, sigma) 

        elif self.guidance == "diffpir":
            hat_x0 = self._diffpir_guidance_impl(x, sigma)

        elif self.guidance == "stsl":
            hat_x0 = self._stsl_guidance_impl(x, sigma)

        elif self.guidance == "dps+mle":
            if sigma < self.mle_sigma_thres:
                hat_x0 = self._type_I_guidance_impl(x, sigma)
            else:
                hat_x0 = self._dps_guidance_impl(x, sigma)

        elif self.guidance == "pgdm+mle":
            if sigma < self.mle_sigma_thres:
                hat_x0 = self._type_I_guidance_impl(x, sigma)
            else:
                hat_x0 = self._pgdm_guidance_impl(x, sigma)

        elif self.guidance == "stsl+mle":
            if sigma < self.mle_sigma_thres:
                hat_x0 = self._type_I_guidance_impl(x, sigma)
            else:
                hat_x0 = self._stsl_guidance_impl(x, sigma)   

        else:
            raise ValueError(f"Invalid guidance type: '{self.guidance}'.")
            
        return hat_x0.clip(-1, 1).detach()
    
    def _auto_type_I_guidance_impl(self, x, sigma):
        x = x.requires_grad_()
        x0_mean, x0_var, theta0_var = self.uncond_pred(x, sigma)
        likelihood_score = grad(self.loglikelihood(x0_mean, x0_var, theta0_var), x)[0]
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score
        return hat_x0

    def _dps_guidance_impl(self, x, sigma):
        assert self.zeta is not None, "zeta must be specified for DPS guidance"
        x = x.requires_grad_()
        x0_mean = self.uncond_pred(x, sigma)[0]
        difference = self.y - self.operator.forward(x0_mean, noiseless=True)
        norm = torch.linalg.norm(difference)
        likelihood_score = -grad(norm, x)[0] * self.zeta
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score
        return hat_x0

    def _pgdm_guidance_impl(self, x, sigma):
        x = x.requires_grad_()
        x0_mean = self.uncond_pred(x, sigma)[0]
        x0_var = sigma.pow(2) / (1 + sigma.pow(2))
        mat = self.mat_solver(self.operator, self.y, x0_mean, x0_var)
        likelihood_score = grad((mat.detach() * x0_mean).sum(), x)[0] * x0_var
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score 
        return hat_x0

    def _diffpir_guidance_impl(self, x, sigma):
        assert self.lambda_ is not None, "lambda_ must be specified for DiffPIR guidance"
        x0_mean = self.uncond_pred(x, sigma)[0]
        x0_var = sigma.pow(2) / self.lambda_
        mat = self.mat_solver(self.operator, self.y, x0_mean, x0_var)
        hat_x0 = x0_mean + mat * x0_var
        return hat_x0

    def _type_I_guidance_impl(self, x, sigma):
        x = x.requires_grad_()
        x0_mean, x0_var, theta0_var = self.uncond_pred(x, sigma)
        mat = self.mat_solver(self.operator, self.y, x0_mean, 
                              x0_var if self.ortho_tf_type is None else theta0_var, self.ortho_tf)
        likelihood_score = grad((mat.detach() * x0_mean).sum(), x)[0]
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score
        return hat_x0
    
    def _type_II_guidance_impl(self, x, sigma):
        x0_mean, x0_var, theta0_var = self.uncond_pred(x, sigma)
        mat = self.mat_solver(self.operator, self.y, x0_mean, 
                              x0_var if self.ortho_tf_type is None else theta0_var, self.ortho_tf)
        ot = self.ortho_tf
        iot = self.ortho_tf.inv
        hat_x0 = x0_mean + iot(ot(mat) * (x0_var if self.ortho_tf_type is None else theta0_var))
        return hat_x0
    
    def _stsl_guidance_impl(self, x, sigma):
        assert self.zeta is not None and self.eta is not None and self.num_hutchinson_samples is not None, \
            "zeta, eta, and num_hutchinson_samples must be specified for STSL guidance"
        x = x.requires_grad_()
        
        # first order loss
        x0_mean = self.uncond_pred(x, sigma)[0]
        difference = self.y - self.operator.forward(x0_mean, noiseless=True)
        first_order_loss = -torch.linalg.norm(difference)
        
        # second order loss
        second_order_loss = 0
        for _ in range(self.num_hutchinson_samples):
            eps = torch.randn_like(x)
            increase_x0_mean = self.uncond_pred(x + eps, sigma)[0]
            second_order_loss += -((increase_x0_mean - x0_mean) * eps).sum() * sigma.pow(2)
        second_order_loss /= self.num_hutchinson_samples
        loss = self.zeta * first_order_loss + (self.eta / x.numel()) * second_order_loss
        
        # approximate E[x0|xt, y]
        likelihood_score = grad(loss, x)[0]
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score 
        
        return hat_x0
        

class ConditionOpenAIDenoiser(ConditionDenoiser):
    
    def __init__(
        self, 
        inner_model, 
        diffusion: GaussianDiffusion, 
        x0_cov_type,
        recon_mse,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.inner_model = inner_model
        self.diffusion = diffusion
        self.denoiser = OpenAIDenoiser(inner_model, diffusion, device=self.device)
        self.x0_cov_type = x0_cov_type
        self.recon_mse = recon_mse
        if recon_mse is not None:
            for key in self.recon_mse.keys():
                self.recon_mse[key] = self.recon_mse[key].to(self.device)

    def uncond_pred(self, x, sigma):
        c_out, c_in = self.denoiser.get_scalings(sigma)
        t = self.denoiser.sigma_to_t(sigma).long()
        D = self.diffusion

        if self.x0_cov_type == 'tmpd':
            x = x.requires_grad_()
        xprev_pred = D.p_mean_variance(self.inner_model, x * c_in, t)
        x0_mean = xprev_pred['pred_xstart']

        if self.x0_cov_type == 'convert':
            if sigma < self.mle_sigma_thres:
                x0_var = (
                    (xprev_pred['variance'] - _extract_into_tensor(D.posterior_variance, t, x.shape)) \
                    / _extract_into_tensor(D.posterior_mean_coef1, t, x.shape).pow(2)
                ).clip(min=1e-6) # Eq. (22)       
            else:
                x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 

        elif self.x0_cov_type == 'analytic':
            assert self.recon_mse is not None
            if sigma < self.mle_sigma_thres:
                idx = (self.recon_mse['sigmas'] - sigma[0]).abs().argmin()
                x0_var = self.recon_mse['mse_list'][idx]
            else:
                x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 

        elif self.x0_cov_type == 'pgdm':
            x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 

        elif self.x0_cov_type == 'dps':
            x0_var = torch.zeros(1).to(self.device) 

        elif self.x0_cov_type == 'diffpir':
            assert self.lambda_ is not None
            x0_var = sigma.pow(2) / self.lambda_ 

        elif self.x0_cov_type == 'tmpd':      
            x0_var = grad(x0_mean.sum(), x, retain_graph=True)[0] * sigma.pow(2)

        else:
            raise ValueError('Invalid posterior covariance type.')
                        
        return x0_mean, x0_var, x0_var


class ConditionOpenAIDenoiserV2(ConditionDenoiser):

    def __init__(self, denoiser: OpenAIDenoiserV2, **kwargs):
        super().__init__(**kwargs)
        self.denoiser = denoiser
        ortho_tf_type = kwargs.get('ortho_tf_type', None)
        if ortho_tf_type is not None:
            assert ortho_tf_type == denoiser.ortho_tf_type, \
                "ortho_tf_type must match the one used in the denoiser"

    def uncond_pred(self, x, sigma):
        c_out, c_in = self.denoiser.get_scalings(sigma)
        model_output, logvar, logvar_ot = self.denoiser(x, sigma, return_variance=True)

        x0_mean = model_output * c_out + x

        if sigma < self.mle_sigma_thres:
            x0_var = logvar.exp() * c_out.pow(2)
            theta0_var = logvar_ot.exp() * c_out.pow(2)
        else:
            x0_var = sigma.pow(2) / (1 + sigma.pow(2))
            theta0_var = sigma.pow(2) / (1 + sigma.pow(2))

        return x0_mean, x0_var, theta0_var
    

#---------------------------------------------
# Implementation of mat solver (computing v)
#---------------------------------------------

__MAT_SOLVER__ = {}


def register_mat_solver(name):
    def wrapper(func):
        __MAT_SOLVER__[name] = func
        return func
    return wrapper


@register_mat_solver('inpainting')
@torch.no_grad()
def inpainting_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    mask = operator.mask
    sigma_s = operator.sigma_s.clip(min=0.001)
    if theta0_var.numel() == 1:
        mat =  (mask * y - mask * x0_mean) / (sigma_s.pow(2) + theta0_var) # TODO
    
    else:
        device = x0_mean.device
        sigma_s, mask, y, x0_mean, theta0_var = \
            sigma_s.cpu(), mask.cpu(), y.cpu(), x0_mean.cpu(), theta0_var.cpu()
        ot = ortho_tf
        iot = ortho_tf.inv

        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (x0_mean.numel(), x0_mean.numel()))

            def _matvec(self, mat):
                mat = torch.Tensor(mat).reshape(x0_mean.shape)
                mat = sigma_s**2 * mat + mask * iot(theta0_var * ot(mat))
                mat = mat.flatten().detach().cpu().numpy()
                return mat
        
        b = (mask * y - mask * x0_mean).flatten().detach().cpu().numpy()
        mat, info = cg(A(), b, tol=1e-4, maxiter=1000)
        if info != 0:
            warn('CG not converge.')
        mat = torch.Tensor(mat).reshape(x0_mean.shape).to(device)
   
    return mat


@torch.no_grad()
def _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if theta0_var.numel() == 1:
        mat = ifft2(fft2(y - ifft2(FB * fft2(x0_mean))) / (sigma_s.pow(2) + theta0_var * F2B) * FBC).real
    
    else:
        device = x0_mean.device
        sigma_s, FB, FBC, F2B, FBFy, y, x0_mean, theta0_var = \
            sigma_s.cpu(), FB.cpu(), FBC.cpu(), F2B.cpu(), FBFy.cpu(), y.cpu(), x0_mean.cpu(), theta0_var.cpu()
        ot = ortho_tf
        iot = ortho_tf.inv

        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (y.numel(), y.numel()))

            def _matvec(self, u):
                u = torch.Tensor(u).reshape(y.shape)
                u = sigma_s**2 * u + ifft2(FB * fft2(iot(theta0_var * ot(ifft2(FBC * fft2(u)).real)))).real
                u = u.flatten().detach().cpu().numpy()
                return u
        
        b = y - ifft2(FB * fft2(x0_mean)).real
        b = b.flatten().detach().cpu().numpy()

        u, info = cg(A(), b, tol=1e-4, maxiter=1000)
        if info != 0:
            warn('CG not converge.')
        u = torch.Tensor(u).reshape(y.shape)

        mat = (ifft2(FBC * fft2(u)).real).to(device)
   
    return mat


@register_mat_solver('gaussian_blur')
@torch.no_grad()
def gaussian_blur_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    return _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf)


@register_mat_solver('motion_blur')
@torch.no_grad()
def motion_blur_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    return _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf)


@register_mat_solver('super_resolution')
@torch.no_grad()
def super_resolution_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    sigma_s = operator.sigma_s.clip(min=0.001).clip(min=1e-2)
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated
    
    if theta0_var.numel() == 1:
        invW = torch.mean(sr.splits(F2B, sf), dim=-1, keepdim=False)
        mat = ifft2(FBC * (fft2(y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)) / (sigma_s.pow(2) + theta0_var * invW)).repeat(1, 1, sf, sf)).real
    
    else:
        device = x0_mean.device
        sigma_s, FB, FBC, F2B, FBFy, y, x0_mean, theta0_var = \
            sigma_s.cpu(), FB.cpu(), FBC.cpu(), F2B.cpu(), FBFy.cpu(), y.cpu(), x0_mean.cpu(), theta0_var.cpu()
        ot = ortho_tf
        iot = ortho_tf.inv

        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (y.numel(), y.numel()))

            def _matvec(self, u):
                u = torch.Tensor(u).reshape(y.shape)
                u = sigma_s**2 * u + sr.downsample(ifft2(FB * fft2(iot(theta0_var * ot(ifft2(FBC * fft2(sr.upsample(u, sf))).real)))), sf)
                u = u.real.flatten().detach().cpu().numpy()
                return u
        
        b = (y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)).real
        b = b.flatten().detach().cpu().numpy()

        u, info = cg(A(), b, tol=1e-4, maxiter=1000)
        if info != 0:
            warn('CG not converge.')
        u = torch.Tensor(u).reshape(y.shape)

        mat = (ifft2(FBC * fft2(sr.upsample(u, sf))).real).to(device)

    return mat

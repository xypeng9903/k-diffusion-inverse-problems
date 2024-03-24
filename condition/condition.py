import torch
from torch import nn
from torch.autograd import grad
from torch.fft import fft2, ifft2
from abc import abstractmethod
from functools import partial
import gpytorch
from gpytorch.distributions import MultivariateNormal

import condition.diffpir_utils.utils_sisr as sr
from .utils import OrthoTransform, LazyOTCovariance
from k_diffusion.external import OpenAIDenoiser, OpenAIDenoiserV2
from guided_diffusion.gaussian_diffusion import GaussianDiffusion, _extract_into_tensor


class LazyLikelihoodCovariance(gpytorch.LinearOperator):

    def _matmul(self: gpytorch.LinearOperator, rhs: torch.Tensor) -> torch.Tensor:
        x0_cov, operator = self._kwargs['x0_cov'], self._kwargs['operator']
        sigma_s = operator.sigma_s
        A = partial(operator.forward, noiseless=True, flatten=True)
        AT = partial(operator.transpose, flatten=True)
        
        rhs = rhs.permute(1, 0)
        rhs = sigma_s**2 * rhs + A(x0_cov(AT(rhs)))[1]
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
        mle_sigma_thres=0.2,
        ortho_tf_type=None
    ):
        super().__init__()
        self.operator = operator
        self.y, self.y_flatten = measurement
        self.guidance = guidance
        self.zeta = zeta
        self.lambda_ = lambda_
        self.device = device
        self.mle_sigma_thres = mle_sigma_thres
        self.ortho_tf_type = ortho_tf_type
        self.ortho_tf = OrthoTransform(ortho_tf_type)
    
        self.mat_solver = __MAT_SOLVER__[operator.name]
        self.proximal_solver = __PROXIMAL_SOLVER__[operator.name]

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
            if sigma < self.mle_sigma_thres:
                hat_x0 = self._auto_type_I_guidance_impl(x, sigma)
            else:
                hat_x0 = self._type_I_guidance_impl(x, sigma)

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

        elif self.guidance == "diffpir+mle":
            if sigma < self.mle_sigma_thres:
                hat_x0 = self._type_II_guidance_impl(x, sigma)
            else:
                hat_x0 = self._diffpir_guidance_impl(x, sigma)
                     
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
        assert self.zeta is not None
        x = x.requires_grad_()
        x0_mean, x0_var, theta0_var = self.uncond_pred(x, sigma)
        difference = self.y - self.operator.forward(x0_mean, noiseless=True)
        norm = torch.linalg.norm(difference)
        likelihood_score = -grad(norm, x)[0] * self.zeta
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score
        return hat_x0

    def _pgdm_guidance_impl(self, x, sigma):
        x = x.requires_grad_()
        x0_mean, x0_var, theta0_var = self.uncond_pred(x, sigma)
        mat = self.mat_solver(self.operator, self.y, x0_mean, sigma.pow(2) / (1 + sigma.pow(2)))
        likelihood_score = grad((mat.detach() * x0_mean).sum(), x)[0] * (sigma.pow(2) / (1 + sigma.pow(2)))
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score 
        return hat_x0

    def _diffpir_guidance_impl(self, x, sigma):
        assert self.lambda_ is not None
        x0_mean, x0_var, theta0_var = self.uncond_pred(x, sigma)
        hat_x0 = self.proximal_solver(self.operator, self.y, x0_mean, sigma.pow(2) / self.lambda_)
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
        hat_x0 = self.proximal_solver(self.operator, self.y, x0_mean, 
                                      x0_var if self.ortho_tf_type is None else theta0_var, self.ortho_tf)
        return hat_x0
        

class ConditionOpenAIDenoiser(ConditionDenoiser):
    
    def __init__(
        self, 
        denoiser, 
        diffusion: GaussianDiffusion, 
        operator, 
        measurement, 
        guidance='I',
        zeta=1,
        x0_cov_type='convert',
        mle_sigma_thres=0.2,
        lambda_=None,
        recon_mse=None, 
        device='cpu'
    ):
        
        super().__init__(
            operator=operator, 
            measurement=measurement, 
            guidance=guidance,
            device=device,
            zeta=zeta,
            lambda_=lambda_,
            mle_sigma_thres=mle_sigma_thres,
            ortho_tf_type=None
        )

        self.denoiser = denoiser
        self.diffusion = diffusion
        self.x0_cov_type = x0_cov_type
        self.openai_denoiser = OpenAIDenoiser(denoiser, diffusion, device=device)

        self.recon_mse = recon_mse
        if recon_mse is not None:
            for key in self.recon_mse.keys():
                self.recon_mse[key] = self.recon_mse[key].to(device)

    def uncond_pred(self, x, sigma):
        c_out, c_in = self.openai_denoiser.get_scalings(sigma)
        t = self.openai_denoiser.sigma_to_t(sigma).long()
        D = self.diffusion

        if self.x0_cov_type == 'tmpd':
            x = x.requires_grad_()
        xprev_pred = D.p_mean_variance(self.denoiser, x * c_in, t)
        x0_mean = xprev_pred['pred_xstart']

        if self.x0_cov_type == 'convert':
            if sigma < self.mle_sigma_thres:
                x0_var = (
                    (xprev_pred['variance'] - _extract_into_tensor(D.posterior_variance, t, x.shape)) \
                    / _extract_into_tensor(D.posterior_mean_coef1, t, x.shape).pow(2)
                ).clip(min=1e-6) # Eq. (22)       
            else:
                if self.lambda_ is not None:
                    x0_var = sigma.pow(2) / self.lambda_ 
                else:
                    x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 

        elif self.x0_cov_type == 'analytic':
            assert self.recon_mse is not None
            if sigma < self.mle_sigma_thres:
                idx = (self.recon_mse['sigmas'] - sigma[0]).abs().argmin()
                x0_var = self.recon_mse['mse_list'][idx]
            else:
                if self.lambda_ is not None:
                    x0_var = sigma.pow(2) / self.lambda_ 
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
            if sigma < self.mle_sigma_thres:
                x0_var = grad(x0_mean.sum(), x, retain_graph=True)[0] * sigma.pow(2)      
            else:
                if self.lambda_ is not None:
                    x0_var = sigma.pow(2) / self.lambda_ 
                else:
                    x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 

        else:
            raise ValueError('Invalid posterior covariance type.')
                        
        return x0_mean, x0_var, x0_var


class ConditionOpenAIDenoiserV2(ConditionDenoiser):

    def __init__(
        self, 
        denoiser: OpenAIDenoiserV2,
        operator, 
        measurement, 
        guidance, 
        device='cpu', 
        zeta=None, 
        lambda_=None, 
        mle_sigma_thres=1, 
        ortho_tf_type=None
    ):

        super().__init__(
            operator=operator, 
            measurement=measurement, 
            guidance=guidance,
            device=device,
            zeta=zeta,
            lambda_=lambda_,
            mle_sigma_thres=mle_sigma_thres,
            ortho_tf_type=ortho_tf_type
        )
        self.denoiser = denoiser
        if ortho_tf_type is not None:
            assert ortho_tf_type == denoiser.ortho_tf_type

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
    

class ConditionImageDenoiserV2(ConditionDenoiser):

    def __init__(
        self, 
        denoiser: OpenAIDenoiserV2,
        operator, 
        measurement, 
        guidance, 
        device='cpu', 
        zeta=None, 
        lambda_=None, 
        mle_sigma_thres=1, 
        ortho_tf_type=None
    ):

        super().__init__(
            operator=operator, 
            measurement=measurement, 
            guidance=guidance,
            device=device,
            zeta=zeta,
            lambda_=lambda_,
            mle_sigma_thres=mle_sigma_thres,
            ortho_tf_type=ortho_tf_type
        )
        self.denoiser = denoiser
        if ortho_tf_type is not None:
            assert ortho_tf_type == denoiser.ortho_tf_type

    def uncond_pred(self, x, sigma):
        c_skip, c_out, c_in = self.denoiser.get_scalings(sigma)
        model_output, logvar, logvar_ot = self.denoiser(x, sigma, return_variance=True)

        x0_mean = model_output * c_out + x * c_skip

        if sigma < self.mle_sigma_thres:
            x0_var = logvar.exp() * c_out.pow(2)
            theta0_var = logvar_ot.exp() * c_out.pow(2)
        else:
            x0_var = sigma.pow(2) / (1 + sigma.pow(2))
            theta0_var = sigma.pow(2) / (1 + sigma.pow(2))

        return x0_mean, x0_var, theta0_var


#----------------------------------------------------------------
# Implementation of mat solver (computing v) for type I guidance
#----------------------------------------------------------------

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
        ot = ortho_tf
        iot = ortho_tf.inv
        b = mask * y - mask * x0_mean

        class A(gpytorch.LinearOperator):

            def _matmul(self, mat):
                mat = mat.reshape(x0_mean.shape)
                mat = sigma_s**2 * mat + mask * iot(theta0_var * ot(mat))
                mat = mat.reshape(-1, 1)
                return mat
            
            def _transpose_nonbatch(self: gpytorch.LinearOperator) -> gpytorch.LinearOperator:
                return self
            
            def _size(self) -> torch.Size:
                return torch.Size([x0_mean.numel(), x0_mean.numel()])
        
        mat = gpytorch.utils.linear_cg(A().matmul, b.reshape(-1, 1), tolerance=1e-4)
        mat = mat.reshape(x0_mean.shape)
   
    return mat


@torch.no_grad()
def _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if theta0_var.numel() == 1:
        mat = ifft2(FBC / (sigma_s.pow(2) + theta0_var * F2B) * fft2(y - ifft2(FB * fft2(x0_mean)))).real
    
    else:
        ot = ortho_tf
        iot = ortho_tf.inv
        b = y - ifft2(FB * fft2(x0_mean)).real

        class A(gpytorch.LinearOperator):

            def _matmul(self, u):
                u = u.reshape(y.shape)
                u = sigma_s**2 * u + ifft2(FB * fft2(iot(theta0_var * ot(ifft2(FBC * fft2(u)).real)))).real
                u = u.reshape(-1, 1)
                return u
            
            def _transpose_nonbatch(self: gpytorch.LinearOperator) -> gpytorch.LinearOperator:
                return self
            
            def _size(self) -> torch.Size:
                return torch.Size([y.numel(), y.numel()])

        u = gpytorch.utils.linear_cg(A().matmul, b.reshape(-1, 1), tolerance=1e-4)
        u = u.reshape(x0_mean.shape) 
        mat = ifft2(FBC * fft2(u)).real
   
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
        ot = ortho_tf
        iot = ortho_tf.inv
        b = (y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)).real

        class A(gpytorch.LinearOperator):

            def _matmul(self, u):
                u = u.reshape(y.shape)
                u = sigma_s**2 * u + sr.downsample(ifft2(FB * fft2(iot(theta0_var * ot(ifft2(FBC * fft2(sr.upsample(u, sf))).real)))), sf)
                u = u.reshape(-1, 1)
                return u
            
            def _transpose_nonbatch(self: gpytorch.LinearOperator) -> gpytorch.LinearOperator:
                return self
            
            def _size(self) -> torch.Size:
                return torch.Size([y.numel(), y.numel()])
        
        u = gpytorch.utils.linear_cg(A().matmul, b.reshape(-1, 1), tolerance=1e-4)
        u = u.reshape(x0_mean.shape)
        mat = ifft2(FBC * fft2(sr.upsample(u, sf))).real

    return mat


#----------------------------------------------------------------------------------
# Implementation of proximal solver (computing E_q[x0|xt, y]) for type II guidance
#----------------------------------------------------------------------------------

__PROXIMAL_SOLVER__ = {}


def register_proximal_solver(name):
    def wrapper(func):
        __PROXIMAL_SOLVER__[name] = func
        return func
    return wrapper


@register_proximal_solver('inpainting')
@torch.no_grad()
def inpainting_proximal(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    mask = operator.mask
    sigma_s = operator.sigma_s.clip(min=0.001)

    if theta0_var.numel() == 1:
        cond_x0_mean = (theta0_var * y + sigma_s**2 * x0_mean) / (sigma_s**2 + theta0_var * mask)
    
    else:
        ot = ortho_tf
        iot = ortho_tf.inv
        b = y / sigma_s**2 + iot(ot(x0_mean) / theta0_var)

        class A(gpytorch.LinearOperator):
            
            def _matmul(self, x):
                x = x.reshape(x0_mean.shape)
                x = mask * x / sigma_s**2 + iot(ot(x) / theta0_var)
                x = x.reshape(-1, 1)
                return x
            
            def _transpose_nonbatch(self: gpytorch.LinearOperator) -> gpytorch.LinearOperator:
                return self
            
            def _size(self) -> torch.Size:
                return torch.Size([x0_mean.numel(), x0_mean.numel()])
            
        cond_x0_mean = gpytorch.utils.linear_cg(A().matmul, b.reshape(-1, 1), initial_guess=x0_mean.reshape(-1, 1), tolerance=1e-4)
        cond_x0_mean = cond_x0_mean.reshape(x0_mean.shape) 
    
    return cond_x0_mean


@torch.no_grad()
def _deblur_proximal(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if theta0_var.numel() == 1:
        rho = sigma_s.pow(2) / theta0_var
        tau = rho.float().repeat(1, 1, 1, 1)
        cond_x0_mean = sr.data_solution(x0_mean.float(), FB, FBC, F2B, FBFy, tau, 1).float()
    
    else:
        ot = ortho_tf
        iot = ortho_tf.inv
        b = ifft2(FBFy).real / sigma_s**2 + iot(ot(x0_mean) / theta0_var)

        class A(gpytorch.LinearOperator):

            def _matmul(self, x):
                x = x.reshape(x0_mean.shape)
                x = ifft2(F2B * fft2(x)).real / sigma_s**2 + iot(ot(x) / theta0_var)
                x = x.reshape(-1, 1)
                return x
            
            def _transpose_nonbatch(self: gpytorch.LinearOperator) -> gpytorch.LinearOperator:
                return self
            
            def _size(self) -> torch.Size:
                return torch.Size([x0_mean.numel(), x0_mean.numel()])
            
        cond_x0_mean = gpytorch.utils.linear_cg(A().matmul, b.reshape(-1, 1), initial_guess=x0_mean.reshape(-1, 1), tolerance=1e-4)
        cond_x0_mean = cond_x0_mean.reshape(x0_mean.shape)
    
    return cond_x0_mean


@register_proximal_solver('gaussian_blur')
@torch.no_grad()
def gaussian_blur_proximal(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    return _deblur_proximal(operator, y, x0_mean, theta0_var, ortho_tf)


@register_proximal_solver('motion_blur')
@torch.no_grad()
def motion_blur_proximal(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    return _deblur_proximal(operator, y, x0_mean, theta0_var, ortho_tf)


@register_proximal_solver('super_resolution')
@torch.no_grad()
def super_resolution_proximal(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform()):
    sigma_s = operator.sigma_s.clip(min=0.001)
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if theta0_var.numel() == 1:
        rho = sigma_s.pow(2) / theta0_var
        tau = rho.float().repeat(1, 1, 1, 1)
        cond_x0_mean = sr.data_solution(x0_mean.float(), FB, FBC, F2B, FBFy, tau, sf).float()

    else:
        ot = ortho_tf
        iot = ortho_tf.inv
        b = ifft2(FBFy).real / sigma_s**2 + iot(ot(x0_mean) / theta0_var)

        class A(gpytorch.LinearOperator):

            def _matmul(self, x):
                x = x.reshape(x0_mean.shape)
                x = ifft2(FBC * fft2(sr.upsample(sr.downsample(ifft2(FB * fft2(x)), sf), sf))).real / sigma_s**2 + iot(ot(x) / theta0_var)
                x = x.reshape(-1, 1)
                return x
            
            def _transpose_nonbatch(self: gpytorch.LinearOperator) -> gpytorch.LinearOperator:
                return self
            
            def _size(self) -> torch.Size:
                return torch.Size([x0_mean.numel(), x0_mean.numel()])
            
        cond_x0_mean = gpytorch.utils.linear_cg(A().matmul, b.reshape(-1, 1), initial_guess=x0_mean.reshape(-1, 1), tolerance=1e-4)
        cond_x0_mean = cond_x0_mean.reshape(x0_mean.shape) 

    return cond_x0_mean

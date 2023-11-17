from torch import nn
from torch.autograd import grad
import torch
from k_diffusion.external import OpenAIDenoiser
from k_diffusion.models import ImageDenoiserModelV2
from guided_diffusion.gaussian_diffusion import GaussianDiffusion, _extract_into_tensor
from k_diffusion import utils
from torch.fft import fft2, ifft2
import condition.diffpir_utils.utils_sisr as sr
from scipy.sparse.linalg import cg, LinearOperator
import numpy as np


class ConditionDenoiser(nn.Module):
    '''Approximate E[x0|xt, y] given denoiser E[x0|xt]'''

    def __init__(
        self,
        operator, 
        measurement, 
        guidance,
        device='cpu',
        zeta=None,
        lambda_=None,
        mle_sigma_thres=0.2
    ):
        super().__init__()
        self.operator = operator
        self.y = measurement
        self.guidance = guidance
        self.zeta = zeta
        self.lambda_ = lambda_
        self.device = device
        self.mle_sigma_thres = mle_sigma_thres
    
        self.mat_solver = __MAT_SOLVER__[operator.name]
        self.proximal_solver = __PROXIMAL_SOLVER__[operator.name]

    def forward(self, x, sigma):
        assert x.shape[0] == 1

        if self.guidance == "dps":
            hat_x0 = self._dps_guidance_impl(x, sigma)

        elif self.guidance == "pgdm":
            hat_x0 = self._pgdm_guidance_impl(x, sigma) 

        elif self.guidance == "diffpir":
            hat_x0 = self._diffpir_guidance_impl(x, sigma)
        
        elif self.guidance == "I":
            hat_x0 = self._type_I_guidance_impl(x, sigma)

        elif self.guidance == "II":
            hat_x0 = self._type_II_guidance_impl(x, sigma)

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
            raise ValueError("Invalid guidance type.")
            
        return hat_x0.clip(-1, 1).detach()

    def _dps_guidance_impl(self, x, sigma):
        assert self.zeta is not None
        x = x.requires_grad_()
        x0_mean = self.uncond_x0_mean_var(x, sigma)['mean']
        difference = self.y - self.operator.forward(x0_mean, noiseless=True)
        norm = torch.linalg.norm(difference)
        likelihood_score = -grad(norm, x)[0] * self.zeta
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score
        return hat_x0

    def _pgdm_guidance_impl(self, x, sigma):
        x = x.requires_grad_()
        x0_mean = self.uncond_x0_mean_var(x, sigma)['mean']
        x0_var = sigma.pow(2) / (1 + sigma.pow(2))
        mat = self.mat_solver(self.operator, self.y, x0_mean, x0_var)
        likelihood_score = grad((mat.detach() * x0_mean).sum(), x)[0]
        hat_x0 = x0_mean + sigma.pow(2) * likelihood_score * x0_var
        return hat_x0

    def _diffpir_guidance_impl(self, x, sigma):
        assert self.lambda_ is not None
        x0_mean = self.uncond_x0_mean_var(x, sigma)['mean']
        x0_var = sigma.pow(2) / self.lambda_
        hat_x0 = self.proximal_solver(self.operator, self.y, x0_mean, x0_var)
        return hat_x0

    def _type_I_guidance_impl(self, x, sigma):
        x = x.requires_grad_()
        uncond_x0_pred = self.uncond_x0_mean_var(x, sigma)
        mat = self.mat_solver(self.operator, self.y, uncond_x0_pred["mean"].detach(), uncond_x0_pred["var"].detach())
        likelihood_score = grad((mat.detach() * uncond_x0_pred["mean"]).sum(), x)[0]
        hat_x0 = uncond_x0_pred["mean"] + sigma.pow(2) * likelihood_score
        return hat_x0

    def _type_II_guidance_impl(self, x, sigma):
        uncond_x0_pred = self.uncond_x0_mean_var(x, sigma)
        hat_x0 = self.proximal_solver(self.operator, self.y, uncond_x0_pred["mean"].detach(), uncond_x0_pred["var"].detach())
        return hat_x0

    def uncond_x0_mean_var(self, x, sigma):
        raise NotImplementedError
    

class ConditionImageDenoiserV2(ConditionDenoiser):
    '''Approximate E[x0|xt, y] given denoiser E[x0|xt]'''
    
    def __init__(
            self, 
            denoiser: ImageDenoiserModelV2, 
            operator, 
            measurement, 
            guidance='I',
            zeta=1,
            x0_cov_type='mle',
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
            mle_sigma_thres=mle_sigma_thres
        )

        self.denoiser = denoiser
        self.x0_cov_type = x0_cov_type
        self.lambda_ = lambda_
        self.mle_sigma_thres = mle_sigma_thres

        self.recon_mse = recon_mse
        if recon_mse is not None:
            for key in self.recon_mse.keys():
                self.recon_mse[key] = self.recon_mse[key].to(device)
    
    def uncond_x0_mean_var(self, x, sigma):
        denoiser_pred = self.denoiser(x, sigma, return_variance=True)
        x0_mean = denoiser_pred[0]

        if self.x0_cov_type == 'mle':
            x0_var = denoiser_pred[1]  

        elif self.x0_cov_type == 'analytic':
            assert self.recon_mse is not None
            idx = (self.recon_mse['sigmas'] - sigma[0]).abs().argmin()
            x0_var = self.recon_mse['mse_list'][idx]

        elif self.x0_cov_type == 'pgdm':
            x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 

        elif self.x0_cov_type == 'dps':
            x0_var = torch.zeros(1).to(self.device) 

        elif self.x0_cov_type == 'diffpir':
            assert self.lambda_ is not None
            x0_var = sigma.pow(2) / self.lambda_ 

        else:
            raise ValueError('Invalid posterior covariance type.')
        
        return {"mean": x0_mean, "var": x0_var}


class ConditionOpenAIDenoiser(ConditionDenoiser):
    '''Approximate E[x0|xt, y] given denoiser E[x0|xt]'''
    
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
            mle_sigma_thres=mle_sigma_thres
        )

        self.denoiser = denoiser
        self.diffusion = diffusion
        self.guidance = guidance
        self.x0_cov_type = x0_cov_type
        self.lambda_ = lambda_
        self.mle_sigma_thres = mle_sigma_thres
        self.device = device

        self.openai_denoiser = OpenAIDenoiser(denoiser, diffusion, device=device)

        self.recon_mse = recon_mse
        if recon_mse is not None:
            for key in self.recon_mse.keys():
                self.recon_mse[key] = self.recon_mse[key].to(device)


    def uncond_x0_mean_var(self, x, sigma):
        c_out, c_in = [utils.append_dims(x, x.ndim) for x in self.openai_denoiser.get_scalings(sigma)]
        t = self.openai_denoiser.sigma_to_t(sigma).long()
        D = self.diffusion

        xprev_pred = D.p_mean_variance(self.denoiser, x * c_in, t)
        x0_mean = xprev_pred['pred_xstart']

        if self.x0_cov_type == 'convert':
            if sigma < self.mle_sigma_thres:
                x0_var = (
                    (xprev_pred['variance'] - _extract_into_tensor(D.posterior_variance, t, x.shape)) \
                    / _extract_into_tensor(D.posterior_mean_coef1, t, x.shape).pow(2)
                ).clip(min=1e-6) # Eq. (22)       
            else:
                if self.guidance == "I":
                    x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 
                elif self.guidance == "II":
                    x0_var = sigma.pow(2) / self.lambda_ 
                else:
                    x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 

        elif self.x0_cov_type == 'analytic':
            assert self.recon_mse is not None
            if sigma < self.mle_sigma_thres:
                idx = (self.recon_mse['sigmas'] - sigma[0]).abs().argmin()
                x0_var = self.recon_mse['mse_list'][idx]
            else:
                if self.guidance == "I":
                    x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 
                elif self.guidance == "II":
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
            
        else:
            raise ValueError('Invalid posterior covariance type.')
                        
        return {"mean": x0_mean, "var": x0_var}


#----------------------------------------------------------------
# Implementation of mat solver (computing v) for type I guidance
#----------------------------------------------------------------

__MAT_SOLVER__ = {}


def register_mat_solver(name):
    def wrapper(func):
        __MAT_SOLVER__[name] = func
        return func
    return wrapper


@register_mat_solver('colorization')
@torch.no_grad()
def colorization_mat(operator, y, x0_mean, x0_var):
    # The form of the solution in colorization is the same for isotrophic and diagnoal posterior covariance
    if x0_var.numel() == 1:
        x0_var = x0_var * torch.ones_like(x0_mean)
    sigma_s = operator.sigma_s.clip(min=0.001)
    mat =  ((y - operator.forward(x0_mean, noiseless=True)) / (sigma_s.pow(2) + x0_var.mean(dim=[1]) / 3)).repeat(1, 3, 1, 1) / 3
    return mat


@register_mat_solver('inpainting')
@torch.no_grad()
def inpainting_mat(operator, y, x0_mean, x0_var):
    # The form of the solution in inpainting is the same for isotrophic and diagnoal posterior covariance
    mask = operator.mask
    sigma_s = operator.sigma_s.clip(min=0.001)
    mat =  (mask * y - mask * x0_mean) / (sigma_s.pow(2) + x0_var)
    return mat


@torch.no_grad()
def _deblur_mat(operator, y, x0_mean, x0_var):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if x0_var.numel() == 1:
        mat = ifft2(FBC / (sigma_s.pow(2) + x0_var * F2B) * fft2(y - ifft2(FB * fft2(x0_mean)))).real
        # Why "mat = ifft2((FBFy - F2B * fft2(x0_mean)) / (sigma_s.pow(2) + x0_var * F2B)).real" has numerical issue?
    else:
        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (y.numel(), y.numel()))

            def _matvec(self, u):
                u = torch.Tensor(u).reshape(y.shape).to(y.device)
                u = sigma_s**2 * u + ifft2(FB * fft2(x0_var * ifft2(FBC * fft2(u))))
                u = u.real.flatten().detach().cpu().numpy()
                return u
        
        b = (y - ifft2(FB * fft2(x0_mean))).real.flatten().detach().cpu().numpy()
        u = cg(A(), b)[0]
        u = torch.Tensor(u.reshape(y.shape)).to(y.device)
        mat = ifft2(FBC * fft2(u)).real
   
    return mat


@register_mat_solver('gaussian_blur')
@torch.no_grad()
def gaussian_blur_mat(operator, y, x0_mean, x0_var):
    return _deblur_mat(operator, y, x0_mean, x0_var)


@register_mat_solver('motion_blur')
@torch.no_grad()
def motion_blur_mat(operator, y, x0_mean, x0_var):
    return _deblur_mat(operator, y, x0_mean, x0_var)


@register_mat_solver('super_resolution')
@torch.no_grad()
def super_resolution_mat(operator, y, x0_mean, x0_var):
    sigma_s = operator.sigma_s.clip(min=0.001).clip(min=1e-2)
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated
    
    if x0_var.numel() == 1:
        invW = torch.mean(sr.splits(F2B, sf), dim=-1, keepdim=False)
        mat = ifft2(FBC * (fft2(y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)) / (sigma_s.pow(2) + x0_var * invW)).repeat(1, 1, sf, sf)).real
    
    else:
        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (y.numel(), y.numel()))

            def _matvec(self, u):
                u = torch.Tensor(u).reshape(y.shape).to(y.device)
                u = sigma_s**2 * u + sr.downsample(ifft2(FB * fft2(x0_var * ifft2(FBC * fft2(sr.upsample(u, sf))))), sf)
                u = u.real.flatten().detach().cpu().numpy()
                return u
        
        b = (y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)).real.flatten().detach().cpu().numpy()
        u = cg(A(), b)[0]
        u = torch.Tensor(u.reshape(y.shape)).to(y.device)
        mat = ifft2(FBC * fft2(sr.upsample(u, sf))).real

    return mat


#----------------------------------------------------------------------------------
# Implementation of proximal solver (computing E_q[x0|xt, y]) for type II guidance
#----------------------------------------------------------------------------------

__PROXIMAL_SOLVER__ = {}


def get_proximal_solver(name):
    return __PROXIMAL_SOLVER__[name]


def register_proximal_solver(name):
    def wrapper(func):
        __PROXIMAL_SOLVER__[name] = func
        return func
    return wrapper


@register_proximal_solver('colorization')
def colorization_proximal(operator, y, x0_mean, x0_var):
    sigma_s = operator.sigma_s.clip(min=0.001)

    if x0_var.numel() == 1:
        rho = (sigma_s.pow(2) / x0_var.mean()).clip(min=0.001)
        d = y.repeat(1, 3, 1, 1) / 3 / rho + x0_mean 
        cond_x0_mean = d - ((d.mean(dim=[1]).repeat(1, 3, 1, 1) / 3) / (1/3 + rho))

    else:
        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (x0_mean.numel(), x0_mean.numel()))

            def _matvec(self, x):
                x = torch.Tensor(x).reshape(x0_mean.shape).to(x0_mean.device)
                x = x / x0_var * sigma_s**2 + (x0_mean.mean(dim=[1]).repeat(1, 3, 1, 1) / 3) 
                x = x.flatten().detach().cpu().numpy()
                return x
            
        b = (y.repeat(1, 3, 1, 1) / 3 + x0_mean / x0_var * sigma_s**2).flatten().detach().cpu().numpy()
        cond_x0_mean = cg(A(), b, x0=x0_mean.flatten().detach().cpu().numpy(), maxiter=50)[0]
        cond_x0_mean = torch.Tensor(cond_x0_mean).reshape(x0_mean.shape).to(x0_mean.device)

    return cond_x0_mean


@register_proximal_solver('inpainting')
def inpainting_proximal(operator, y, x0_mean, x0_var):
    # The form of the solution in inpainting is the same for isotrophic and diagnoal posterior covariance
    mask = operator.mask
    sigma_s = operator.sigma_s.clip(min=0.001)
    return (x0_var * mask * y + sigma_s**2 * x0_mean) / (x0_var * mask + sigma_s**2)


def _deblur_proximal(operator, y, x0_mean, x0_var):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if x0_var.numel() == 1:
        rho = sigma_s.pow(2) / x0_var.mean()
        tau = rho.float().repeat(1, 1, 1, 1)
        cond_x0_mean = sr.data_solution(x0_mean.float(), FB, FBC, F2B, FBFy, tau, 1).float()
    
    else:
        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (x0_mean.numel(), x0_mean.numel()))

            def _matvec(self, x):
                x = torch.Tensor(x).reshape(x0_mean.shape).to(x0_mean.device)
                x = sigma_s**2 * x + x0_var * ifft2(F2B * fft2(x))
                x = x.real.flatten().detach().cpu().numpy()
                return x
            
        b = (x0_var * ifft2(FBFy) + sigma_s**2 * x0_mean).real.flatten().detach().cpu().numpy()
        cond_x0_mean = cg(A(), b, x0=x0_mean.flatten().detach().cpu().numpy())[0]
        cond_x0_mean = torch.Tensor(cond_x0_mean).reshape(x0_mean.shape).to(x0_mean) 
    
    return cond_x0_mean


@register_proximal_solver('gaussian_blur')
def gaussian_blur_proximal(operator, y, x0_mean, x0_var):
    return _deblur_proximal(operator, y, x0_mean, x0_var)


@register_proximal_solver('motion_blur')
def motion_blur_proximal(operator, y, x0_mean, x0_var):
    return _deblur_proximal(operator, y, x0_mean, x0_var)


@register_proximal_solver('super_resolution')
def super_resolution_proximal(operator, y, x0_mean, x0_var):
    sigma_s = operator.sigma_s.clip(min=0.001)
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if x0_var.numel() == 1:
        rho = sigma_s.pow(2) / x0_var.mean()
        tau = rho.float().repeat(1, 1, 1, 1)
        cond_x0_mean = sr.data_solution(x0_mean.float(), FB, FBC, F2B, FBFy, tau, sf).float()

    else:
        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (x0_mean.numel(), x0_mean.numel()))

            def _matvec(self, x):
                x = torch.Tensor(x).reshape(x0_mean.shape).to(x0_mean.device)
                x = sigma_s**2 * x + x0_var * ifft2(FBC * fft2(sr.upsample(sr.downsample(ifft2(FB * fft2(x)), sf), sf)))
                x = x.real.flatten().detach().cpu().numpy()
                return x
            
        b = (x0_var * ifft2(FBFy) + sigma_s**2 * x0_mean).real.flatten().detach().cpu().numpy()
        cond_x0_mean = cg(A(), b, x0=x0_mean.flatten().detach().cpu().numpy())[0]
        cond_x0_mean = torch.Tensor(cond_x0_mean).reshape(x0_mean.shape).to(x0_mean.device)

    return cond_x0_mean


from torch import nn
from torch.autograd import grad
import torch
from k_diffusion.external import OpenAIDenoiser
from guided_diffusion.gaussian_diffusion import GaussianDiffusion, _extract_into_tensor
from k_diffusion import utils
from torch.fft import fft2, ifft2
import condition.utils.utils_sisr as sr


class ConditionOpenAIDenoiser(OpenAIDenoiser):
    '''Approximate E[x0|xt, y] given denoiser E[x0|xt]'''
    
    def __init__(
            self, 
            model, 
            diffusion: GaussianDiffusion, 
            operator, 
            measurement, 
            guidance='I',
            xstart_cov_type='mle',
            lambda_ = None,
            recon_mse = None,
            quantize=False, 
            has_learned_sigmas=True, 
            device='cpu' 
    ):
        super().__init__(model, diffusion, quantize, has_learned_sigmas, device)
        self.diffusion = diffusion
        self.operator = operator
        self.y = measurement
        self.guidance = guidance
        self.xstart_cov_type = xstart_cov_type
        self.lambda_ = lambda_
        self.recon_mse = recon_mse
        for key in self.recon_mse.keys():
            self.recon_mse[key] = self.recon_mse[key].to(device)
        self.mat_solver = __MAT_SOLVER__[operator.name]
        self.proximal_solver = __PROXIMAL_SOLVER__[operator.name]

    def forward(self, x: torch.Tensor, sigma):
        assert x.shape[0] == 1
        if self.guidance == "I":
            x = x.requires_grad_()
            uncond_xstart_pred = self.uncond_xstart_mean_variance(x, sigma)
            return self._type_I_guidance(
                uncond_xstart_pred["mean"],
                uncond_xstart_pred["xstart_cov"],
                x,
                sigma
            )
        elif self.guidance == "II":
            uncond_xstart_pred = self.uncond_xstart_mean_variance(x, sigma)
            return self._type_II_guidance(
                uncond_xstart_pred["mean"],
                uncond_xstart_pred["xstart_cov"]
            )
        else:
            raise NotImplementedError
    
    def uncond_xstart_mean_variance(self, x, sigma):
        c_out, c_in = [utils.append_dims(x, x.ndim) for x in self.get_scalings(sigma)]
        t = self.sigma_to_t(sigma).long()
        D = self.diffusion

        xprev_pred = D.p_mean_variance(self.inner_model, x * c_in, t)
        xstart_mean = xprev_pred['pred_xstart']

        if self.xstart_cov_type == 'mle':
            xstart_cov = (
                (xprev_pred['variance'] - _extract_into_tensor(D.posterior_variance, t, x.shape)) \
                / _extract_into_tensor(D.posterior_mean_coef1, t, x.shape).pow(2)
            ).clip(min=0)
            if self.recon_mse is not None:
                idx = (self.recon_mse['sigmas'] - sigma[0]).abs().argmin()
                variance = self.recon_mse['mse_list'][idx]
            else:
                variance = sigma.pow(2) / (1 + sigma.pow(2))
            xstart_cov = xstart_cov if sigma < 0.3 \
                                    else variance * torch.ones_like(xstart_mean)
        elif self.xstart_cov_type == 'pgdm':
            xstart_cov = sigma.pow(2) / (1 + sigma.pow(2)) * torch.ones_like(xstart_mean)
        elif self.xstart_cov_type == 'dps':
            xstart_cov = 0 * torch.ones_like(xstart_mean)
        elif self.xstart_cov_type == 'diffpir':
            assert self.lambda_ is not None
            xstart_cov = sigma.pow(2) / self.lambda_ * torch.ones_like(xstart_mean)
        elif self.xstart_cov_type == 'analytic':
            assert self.recon_mse is not None
            idx = (self.recon_mse['sigmas'] - sigma[0]).abs().argmin()
            xstart_cov = self.recon_mse['mse_list'][idx] * torch.ones_like(xstart_mean)
        else:
            raise NotImplementedError
        
        return {
            "mean": xstart_mean,
            "xstart_cov": xstart_cov
        }

    def _type_I_guidance(self, xstart_mean, xstart_cov, x, sigma):
        assert x.shape[0] == 1
        mat = self.mat_solver(self.operator, self.y, xstart_mean, xstart_cov)
        likelihood_score = grad((mat.detach() * xstart_mean).sum(), x)[0]
        xstart_mean = xstart_mean + sigma.pow(2) * likelihood_score
        return xstart_mean.clip(-1, 1).detach()

    def _type_II_guidance(self, xstart_mean, xstart_cov):
        xstart_mean = self.proximal_solver(self.operator, self.y, xstart_mean, xstart_cov.mean())
        return xstart_mean.clip(-1, 1).detach()


__MAT_SOLVER__ = {}


def register_mat_solver(name):
    def wrapper(func):
        __MAT_SOLVER__[name] = func
        return func
    return wrapper


@register_mat_solver('inpainting')
def inpainting_mat(operator, y, mean_x0, cov_x0):
    mask = operator.mask
    sigma_s = operator.sigma_s
    mat =  (mask * y - mask * mean_x0) / (sigma_s.pow(2) + cov_x0)
    return mat


def _deblur_mat(operator, y, mean_x0, cov_x0):
    sigma_s = operator.sigma_s
    FB, FBC, F2B, FBFy = operator.pre_calculated
    mat = ifft2((FBFy - F2B * fft2(mean_x0)) / (sigma_s.pow(2) + cov_x0 * F2B)).real
    return mat


@register_mat_solver('gaussian_blur')
def gaussian_blur_mat(operator, y, mean_x0, cov_x0):
    return _deblur_mat(operator, y, mean_x0, cov_x0)


@register_mat_solver('motion_blur')
def motion_blur_mat(operator, y, mean_x0, cov_x0):
    return _deblur_mat(operator, y, mean_x0, cov_x0)


@register_mat_solver('super_resolution')
def super_resolution_mat(operator, y, mean_x0, cov_x0):
    sigma_s = operator.sigma_s
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated
    invW = torch.mean(sr.splits(cov_x0 * F2B, sf), dim=-1, keepdim=False)
    mat = ifft2(FBC * (fft2(y - sr.downsample(ifft2(FB * fft2(mean_x0)), sf)) / (sigma_s.pow(2) + invW)).repeat(1, 1, sf, sf)).real
    return mat


# TODO: support diagnol posterior covariance
__PROXIMAL_SOLVER__ = {}


def register_proximal_solver(name):
    def wrapper(func):
        __PROXIMAL_SOLVER__[name] = func
        return func
    return wrapper


@register_proximal_solver('inpainting')
def inpainting_proximal(operator, y, mean_x0, cov_x0):
    mask = operator.mask
    sigma_s = operator.sigma_s
    rho = sigma_s.pow(2) / cov_x0
    return (mask * y + rho * mean_x0) / (mask + rho)


def _deblur_proximal(operator, y, mean_x0, cov_x0):
    sigma_s = operator.sigma_s
    FB, FBC, F2B, FBFy = operator.pre_calculated
    rho = sigma_s.pow(2) / cov_x0
    tau = rho.float().repeat(1, 1, 1, 1)
    return sr.data_solution(mean_x0.float(), FB, FBC, F2B, FBFy, tau, 1).float()


@register_proximal_solver('gaussian_blur')
def gaussian_blur_proximal(operator, y, mean_x0, cov_x0):
    return _deblur_proximal(operator, y, mean_x0, cov_x0)


@register_proximal_solver('motion_blur')
def motion_blur_proximal(operator, y, mean_x0, cov_x0):
    return _deblur_proximal(operator, y, mean_x0, cov_x0)


@register_proximal_solver('super_resolution')
def super_resolution_proximal(operator, y, mean_x0, cov_x0):
    sigma_s = operator.sigma_s
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated
    rho = sigma_s.pow(2) / cov_x0
    tau = rho.float().repeat(1, 1, 1, 1)
    return sr.data_solution(mean_x0.float(), FB, FBC, F2B, FBFy, tau, sf).float()



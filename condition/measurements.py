'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
import torch
import torchvision
import linear_operator
from einops import rearrange

from condition.dps_utils.resizer import Resizer
from condition.dps_utils.img_utils import Blurkernel, fft2_m
from condition.diffpir_utils.utils_sisr import pre_calculate

import numpy as np
from torch.fft import fft2, ifft2
import os
import hdf5storage


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        cls.name = name
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


def convert_to_linear_op(operator):    
    A = OperatorWrapper(torch.zeros((1,), device=operator.device), operator)
    AT = A.transpose(-2, -1)
    sigma_s = operator.sigma_s
    return A, AT, sigma_s    
    
    
class OperatorWrapper(linear_operator.LinearOperator):
                
        def __init__(self, tmp: torch.Tensor, operator, mode=1):
            super().__init__(tmp, operator=operator, mode=mode)
            self.tmp = tmp
            self.operator = operator
            self.mode = mode
        
        def _A(self, rhs):
            c, h, w = self.operator.in_shape[-3:]
            b = rhs.shape[-1]
            rhs = rearrange(rhs, '(c h w) b -> b c h w', b=b, c=c, h=h, w=w)
            rhs = self.operator.forward(rhs, flatten=True, noiseless=True)[1]
            rhs = rhs.transpose(0, 1)
            return rhs
        
        def _AT(self, rhs):
            rhs = rhs.transpose(0, 1)
            rhs = self.operator.transpose(rhs, flatten=True)
            rhs = rhs.reshape(rhs.shape[0], -1)
            rhs = rhs.transpose(0, 1)
            return rhs
        
        def _matmul(self: linear_operator.LinearOperator, rhs: F.Tensor) -> F.Tensor:
            if self.mode == 1:
                return self._A(rhs)
            else:
                return self._AT(rhs)
        
        def _size(self) -> torch.Size:
            in_dim = torch.tensor(self.operator.in_shape).prod().item()
            out_dim = torch.tensor(self.operator.out_shape).prod().item()
            if self.mode == 1:
                return torch.Size([out_dim, in_dim])
            else:
                return torch.Size([in_dim, out_dim])
        
        def _transpose_nonbatch(self) -> linear_operator.LinearOperator:
            args = self._args
            kwargs = self._kwargs
            kwargs['mode'] = -kwargs['mode']
            out = self.__class__(*args, **kwargs)
            return out
        

class LinearOperator(ABC):

    @abstractmethod
    def forward(self, data, flatten=False, noiseless=False):
        raise NotImplementedError("The class {} requires a forward function!".format(self.__class__.__name__))

    def auto_transpose(self, y, flatten=False):
        input = torch.randn(y.shape[0], *self.in_shape[-3:]).to(self.device).requires_grad_()
        res = torch.autograd.grad((y * self.forward(input, flatten=flatten, noiseless=True)).sum(), 
                                  input, retain_graph=True)[0]
        return res


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data
    

@register_operator(name='colorization')
class ColorizationOperator(LinearOperator):
    def __init__(self, sigma_s, device):
        self.device = device
        self.sigma_s = torch.Tensor([sigma_s]).to(device)
        
    def forward(self, data, **kwargs):
        y = data.mean(dim=1, keepdim=True)
        if not kwargs.get('noiseless', False):
            y += self.sigma_s * torch.randn_like(y)
        return y


@register_operator(name='super_resolution')     
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, sigma_s, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)
        self.scale_factor = scale_factor
        self.sigma_s = torch.Tensor([sigma_s]).to(device)

        kernels = hdf5storage.loadmat(os.path.join('condition', 'kernels', 'kernels_bicubicx234.mat'))['kernels']
        k_index = scale_factor - 2 if scale_factor < 5 else 2
        self.kernel = torch.Tensor(kernels[0, k_index].astype(np.float64))

        self.in_shape = in_shape
        out_shape = tuple(int(s / scale_factor) for s in in_shape[-2:])
        self.out_shape = (1, 3, *out_shape)

    def forward(self, data, flatten=False, noiseless=False):
        y = self.down_sample(data) 
        if not noiseless:
            y += self.sigma_s * torch.randn_like(y)
        if flatten:
            return y, y.reshape(y.shape[0], -1)
        return y
    
    def transpose(self, y, flatten=False):
        if flatten:
            y = y.reshape(y.shape[0], *self.out_shape[-3:])
        k = self.get_kernel().to(self.device)
        FB, FBC, F2B, FBFy = pre_calculate(y, k, self.scale_factor)
        x = ifft2(FBFy).real
        return x
    
    def get_kernel(self):
        return self.kernel.view(1, 1, *self.kernel.shape).to(self.device)
    

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, in_shape, kernel_size, intensity, sigma_s, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?
        self.kernel = np.load('./condition/kernels/motion_ks61_std0.5.npy')
        self.conv.update_weights(self.kernel)
        self.sigma_s = torch.Tensor([sigma_s]).to(device)
        self.in_shape = in_shape
        self.out_shape = in_shape

    def forward(self, data, flatten=False, noiseless=False): # TODO
        k = self.get_kernel().to(self.device)
        FB, FBC, F2B, _ = pre_calculate(data, k, 1)
        y = ifft2(FB * fft2(data)).real
        if not noiseless:
            y += self.sigma_s * torch.randn_like(y)
        self.pre_calculated = (FB, FBC, F2B, FBC * fft2(y))
        if flatten:
            return y, y.reshape(y.shape[0], -1)
        return y

    def transpose(self, y, flatten=False): 
        if flatten:
            y = y.reshape(y.shape[0], *self.in_shape[-3:])
        k = self.get_kernel().to(self.device)
        FB, FBC, F2B, _ = pre_calculate(y, k, 1)
        x = ifft2(FBC * fft2(y)).real
        return x
    
    def get_kernel(self):
        kernel = torch.Tensor(self.kernel).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, in_shape, kernel_size, intensity, sigma_s, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        # self.kernel = self.conv.get_kernel()
        self.kernel = torch.Tensor(np.load('./condition/kernels/gaussian_ks61_std3.0.npy')).to(device)
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.sigma_s = torch.Tensor([sigma_s]).to(device)
        self.in_shape = in_shape
        self.out_shape = in_shape

    def forward(self, data, flatten=False, noiseless=False):
        k = self.get_kernel().to(self.device)
        FB, FBC, F2B, _ = pre_calculate(data, k, 1)
        y = ifft2(FB * fft2(data)).real
        if not noiseless:
            y += self.sigma_s * torch.randn_like(y)
        if flatten:
            return y, y.reshape(y.shape[0], -1)
        return y
    
    def transpose(self, y, flatten=False):
        if flatten:
            y = y.reshape(y.shape[0], *self.in_shape[-3:])
        k = self.get_kernel().to(self.device)
        FB, FBC, F2B, _ = pre_calculate(y, k, 1)
        x = ifft2(FBC * fft2(y)).real
        return x
    
    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device, sigma_s, mask):
        self.device = device
        self.sigma_s = torch.Tensor([sigma_s]).to(device)
        self.mask = (torchvision.io.read_image(mask) / 255).round().int().to(self.device)
        self.in_shape = self.mask.shape
        self.out_shape = (1, 1, self.mask.sum())
    
    def forward(self, data: torch.Tensor, flatten=False, noiseless=False):        
        y = data.clone()
        if not noiseless:
            y += self.sigma_s * torch.randn_like(y)
        y = y * self.mask

        if flatten:
            sample_indices = torch.where(self.mask > 0)
            return y, y[..., sample_indices[-3], sample_indices[-2], sample_indices[-1]]

        else:
            '''
            Compute D^T (Dx + n) to address vary-dimensionality, 
            which is equivalent to m \odot (x + n)
            ''' 
            return y
    
    def transpose(self, data, flatten=False):
        y = data.clone()
        if flatten:
            sample_indices = torch.where(self.mask > 0)
            x = torch.zeros(y.shape[0], *self.in_shape[-3:]).to(self.device)
            x[..., sample_indices[-3], sample_indices[-2], sample_indices[-1]] = y
        else:
            x = y
        return x
    

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)
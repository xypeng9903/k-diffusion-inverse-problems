'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel

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


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    def transpose(self, y):
        ones = torch.ones_like(y)
        return torch.autograd.functional.vjp(
            partial(self.forward, noisess=True), ones, y
        )[1]
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


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

    def forward(self, data, **kwargs):
        y = self.down_sample(data) 
        if not kwargs.get('noiseless', False):
            y += self.sigma_s * torch.randn_like(y)
        k = self.get_kernel().to(self.device)
        self.pre_calculated = pre_calculate(y, k, self.scale_factor)
        return y

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    
    def get_kernel(self):
        return self.kernel.view(1, 1, *self.kernel.shape)
    

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, sigma_s, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?
        self.kernel = np.load('./condition/kernels/motion_ks61_std0.5.npy')
        self.conv.update_weights(self.kernel)
        self.sigma_s = torch.Tensor([sigma_s]).to(device)

    def forward(self, data, **kwargs):
        k = self.get_kernel().to(self.device)
        FB, FBC, F2B, _ = pre_calculate(data, k, 1)
        y = ifft2(FB * fft2(data)).real
        if not kwargs.get('noiseless', False):
            y += self.sigma_s * torch.randn_like(y)
        self.pre_calculated = (FB, FBC, F2B, FBC * fft2(y))
        return y

    def get_kernel(self):
        # kernel = torch.Tensor(self.kernel.kernelMatrix).to(self.device)
        kernel = torch.Tensor(self.kernel).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, sigma_s, device):
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

    def forward(self, data, **kwargs):
        k = self.get_kernel().to(self.device)
        FB, FBC, F2B, _ = pre_calculate(data, k, 1)
        y = ifft2(FB * fft2(data)).real
        if not kwargs.get('noiseless', False):
            y += self.sigma_s * torch.randn_like(y)
        self.pre_calculated = (FB, FBC, F2B, FBC * fft2(y))
        return y
    
    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device, sigma_s, mask_opt):
        self.device = device
        self.sigma_s = torch.Tensor([sigma_s]).to(device)
        self.in_shape = (1, 3, mask_opt['image_size'], mask_opt['image_size'])
        self.mask = self.generate_mask(mask_opt)
    
    def forward(self, data: torch.Tensor, **kwargs):        
        '''
            Compute D^T (Dx + n) to address vary-dimensionality, 
            which is equivalent to m \odot (x + n)
        '''
        y = data.clone()
        if not kwargs.get('noiseless', False):
            y += self.sigma_s * torch.randn_like(y)
        return y * self.mask
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
    
    def generate_mask(self, mask_opt):
        mask_generator = MaskGenerator(**mask_opt)
        img = torch.randn(self.in_shape).to(self.device)
        mask = mask_generator(img)
        return mask
    

class MaskGenerator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = self._random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask
    
    def _random_sq_bbox(self, img, mask_shape, image_size=256, margin=(16, 16)):
        """Generate a random sqaure mask for inpainting
        """
        B, C, H, W = img.shape
        h, w = mask_shape
        margin_height, margin_width = margin
        maxt = image_size - margin_height - h
        maxl = image_size - margin_width - w

        # bb
        # t = np.random.randint(margin_height, maxt)
        # l = np.random.randint(margin_width, maxl)
        t = (margin_height + maxt) // 2
        l = (margin_width + maxl) // 2

        # make mask
        mask = torch.ones([B, C, H, W], device=img.device)
        mask[..., t:t+h, l:l+w] = 0

        return mask, t, t+h, l, l+w


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
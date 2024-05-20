import numpy as np
import torch
from torch.autograd import grad
from scipy.fft import dctn, idctn
import pywt
from abc import abstractmethod
from einops import rearrange
from linear_operator.operators import (
    LinearOperator,
    DiagLinearOperator,
    ConstantDiagLinearOperator
)

from .measurements import convert_to_linear_op


#------------------
# Image transforms
#------------------

class ImageTransform(LinearOperator):
            
    def __init__(self, img_size, mode=1):
        super().__init__(img_size=img_size, mode=mode)
        self.img_size = img_size
        self.mode = mode
    
    @abstractmethod
    def _encode(self, x: torch.Tensor):
        pass
    
    @abstractmethod
    def _decode(self, x: torch.Tensor):
        pass
    
    def _matmul(self: LinearOperator, rhs: torch.Tensor) -> torch.Tensor:
        c, h, w = self.img_size
        b = rhs.shape[1]
        rhs = rearrange(rhs, '(c h w) b -> b c h w', b=b, c=c, h=h, w=w)
        if self.mode == 1:
            rhs = self._encode(rhs)
        else:
            rhs = self._decode(rhs)
        rhs = rearrange(rhs, 'b c h w -> (c h w) b', b=b, c=c, h=h, w=w)
        return rhs
    
    def _size(self) -> torch.Size:
        img_dim = torch.tensor(self.img_size).prod().item()
        return torch.Size([img_dim, img_dim])
    
    def _transpose_nonbatch(self: LinearOperator) -> LinearOperator:
        args = self._args
        kwargs = self._kwargs
        kwargs['mode'] = -kwargs['mode']
        out = self.__class__(*args, **kwargs)
        return out
    
    def logdet(self: LinearOperator) -> torch.Tensor:
        return torch.tensor([0])


class DiscreteCosineTransform(ImageTransform):
    
    def _encode(self, x):
        device = x.device
        x = x.detach().cpu().numpy()
        x = dctn(x, norm='ortho')
        x = torch.tensor(x, device=device)
        return x
    
    def _decode(self, x):
        device = x.device
        x = x.detach().cpu().numpy()
        x = idctn(x, norm='ortho')
        x = torch.tensor(x, device=device)
        return x
    
    
class DiscreteWaveletTransform(ImageTransform):
    
    wt_slice = None
    
    def _encode(self, x):
        device = x.device
        x = x.detach().cpu().numpy()
        x = pywt.wavedec2(x, wavelet='haar', level=3, axes=(-2, -1))
        x, wt_slice = pywt.coeffs_to_array(x, axes=(-2, -1))
        self.wt_slice = wt_slice
        x = torch.tensor(x, device=device)
        return x
    
    def _decode(self, x):
        device = x.device
        x = x.detach().cpu().numpy()
        wt_slice = self._get_slice(x)
        x = pywt.array_to_coeffs(x, wt_slice, output_format='wavedec2')
        x = pywt.waverec2(x, wavelet='haar', axes=(-2, -1))
        x = torch.tensor(x, device=device)
        return x
    
    def _get_slice(self, x: np.array): # TODO: can we remove this?
        if self.wt_slice is None:
            x = pywt.wavedec2(x, wavelet='haar', level=3, axes=(-2, -1))
            _, wt_slice = pywt.coeffs_to_array(x, axes=(-2, -1))
            self.wt_slice = wt_slice
        return self.wt_slice


#--------------
# Covariances
#--------------
    
class DiagonalCovariance(DiagLinearOperator):
    
    def __init__(self, variance: torch.Tensor) -> None:
        super().__init__(variance)
        self.variance = variance
    
    
class IsotropicCovariance(ConstantDiagLinearOperator):
    
    def __init__(self, variance: torch.Tensor, diag_shape) -> None:
        super().__init__(variance.reshape(1), diag_shape=diag_shape)
        self.variance = variance
        
                
class TweedieCovariance(LinearOperator):
    
    def __init__(
            self, 
            xt: torch.Tensor, 
            denoiser, 
            sigma: torch.Tensor
        ):
        r"""
        Args:
            denoiser: E[x0|xt]
            xt, sigma: xt = x0 + sigma * n, n ~ N(0, I) 
        """
        super().__init__(xt, denoiser=denoiser, sigma=sigma)
        self.saved = xt, denoiser, sigma
        
    def _matmul(self: LinearOperator, rhs: torch.Tensor) -> torch.Tensor:
        xt, denoiser, sigma = self.saved
        b, c, h, w = xt.shape
        rhs = rearrange(rhs, '(c h w) b -> b c h w', b=b, c=c, h=h, w=w)
        rhs = self._jvp(rhs) * sigma**2
        rhs = rearrange(rhs, 'b c h w -> (c h w) b', b=b, c=c, h=h, w=w)
        return rhs
        
    def _size(self) -> torch.Size:
        xt, denoiser, sigma = self.saved
        x_dim = torch.tensor(xt.shape[-3:]).prod().item()
        return torch.Size([x_dim, x_dim])
    
    def _transpose_nonbatch(self: LinearOperator) -> LinearOperator:
        return self
    
    @torch.enable_grad()
    def _jvp(self, x: torch.Tensor):
        xt, denoiser, sigma = self.saved
        xt = xt.requires_grad_()
        return grad((denoiser(xt, sigma) * x.detach()).sum(), xt)[0] 
    
    
class LikelihoodCovariance(LinearOperator):
    
    def __init__(self, y_flatten, operator, x0_covar):
        super().__init__(y_flatten, operator=operator, x0_covar=x0_covar)
        self.saved = operator, x0_covar
        self.y_flatten = y_flatten
        
    def _matmul(self: LinearOperator, rhs: torch.Tensor) -> torch.Tensor:
        operator, x0_covar = self.saved
        A, AT, sigma_s = convert_to_linear_op(operator)
        return A @ x0_covar @ AT @ rhs + rhs * sigma_s**2
    
    def _size(self) -> torch.Size:
        return torch.Size([self.y_flatten.numel(), self.y_flatten.numel()])
    
    def _transpose_nonbatch(self: LinearOperator) -> LinearOperator:
        return self
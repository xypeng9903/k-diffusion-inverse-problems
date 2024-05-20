from __future__ import annotations

from abc import abstractmethod
import torch
from scipy.fft import dctn, idctn
import pywt
import copy
import numpy as np


class AbstractLinearFunction:
    r"""
    Base class for function from Tensor to Tensor satisfying:
     1) f(x + y) = f(x) + f(y)
     2) f(ax) = af(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The class {} requires a _forward function!".format(self.__class__.__name__))
    
    @abstractmethod
    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The class {} requires a _transpose function!".format(self.__class__.__name__))
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        linear_func = LinearFunction.apply
        return linear_func(x, self)
    

class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, linear_func):
        output = linear_func.forward(input)
        ctx.linear_func = linear_func
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        linear_func = ctx.linear_func
        grad_input = linear_func.transpose(grad_output)
        return grad_input, None
    

#----------------------
# Orthogonal transform
#----------------------

class OrthoTransform:
    def __init__(self, ortho_tf_type=None):
        self.ortho_tf_type = ortho_tf_type
        if ortho_tf_type is not None:
            self.ot = __OT__[ortho_tf_type]()
            self.iot = self.ot.inv()

    def __call__(self, x: torch.Tensor):
        if self.ortho_tf_type is None:
            return x
        else:
            return self.ot(x)
        
    def inv(self, x: torch.Tensor):
        if self.ortho_tf_type is None:
            return x        
        else:
            return self.iot(x)


__OT__ = dict()
          

def register_ot(name: str):
    def wrapper(cls):
        __OT__[name] = cls
        return cls
    return wrapper


class OrthoLinearFunction(AbstractLinearFunction):

    def inv(self) -> OrthoLinearFunction:
        out = copy.deepcopy(self)
        out.forward, out.transpose = out.transpose, out.forward
        return out


@register_ot('dct')
class DiscreteCosineTransform(OrthoLinearFunction):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = dctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        return x

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = idctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        return x
    

@register_ot('dwt')
class DiscreteWaveletTransform(OrthoLinearFunction):
    
    dwt_slice = None

    def __init__(self, level=3, wavelet='haar') -> None:
        super().__init__()
        self.level = level
        self.wavelet = wavelet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = pywt.wavedec2(x, wavelet=self.wavelet, level=self.level, axes=(-2, -1))
        x, dwt_slice = pywt.coeffs_to_array(x, axes=(-2, -1))
        self.dwt_slice = dwt_slice
        x = torch.tensor(x, device=device)
        return x
    
    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        dwt_slice = self._get_slice(x)
        x = pywt.array_to_coeffs(x, dwt_slice, output_format='wavedec2')
        x = pywt.waverec2(x, wavelet=self.wavelet, axes=(-2, -1))
        x = torch.tensor(x, device=device)
        return x

    def _get_slice(self, x: np.array): # TODO: can we remove this?
        if self.dwt_slice is None:
            x = pywt.wavedec2(x, wavelet='haar', level=3, axes=(-2, -1))
            _, dwt_slice = pywt.coeffs_to_array(x, axes=(-2, -1))
            self.dwt_slice = dwt_slice
        return self.dwt_slice


#----------------------
# Posterior covariance
#----------------------

class LazyOTCovariance(AbstractLinearFunction):
    r"""
    Covariance with the form C = W @ diag(v) @ W.T for some "OrthoTransform" W.T 
    """

    def __init__(self, ortho_tf: OrthoTransform, variance: torch.Tensor) -> None:
        super().__init__()
        self.ortho_tf = ortho_tf
        self.variance = variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ortho_tf(x)
        x = x * self.variance
        x = self.ortho_tf.inv(x)
        return x
    
    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
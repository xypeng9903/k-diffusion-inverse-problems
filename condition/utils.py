from __future__ import annotations

from abc import abstractmethod
import torch
from scipy.fft import dctn, idctn
from einops.layers.torch import Rearrange


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
    

class OrthoTransform:
    def __init__(self, ortho_tf_type=None):
        self.ortho_tf_type = ortho_tf_type

    def __call__(self, x: torch.Tensor):
        if self.ortho_tf_type is None:
            return x
        
        elif self.ortho_tf_type == 'dct':
            dct = DCT()
            x = dct(x)
            return x
        
        elif self.ortho_tf_type == 'dct8x8':
            dct = PatchwiseDCT(8, 8)
            x = dct(x)
            return x
        
        else:
            raise ValueError('Invalid transform type')
        
    def inv(self, x: torch.Tensor):
        if self.ortho_tf_type is None:
            return x
        
        elif self.ortho_tf_type == 'dct':
            idct = IDCT()
            x = idct(x)
            return x

        elif self.ortho_tf_type == 'dct8x8':
            idct = PatchwiseIDCT(8, 8)
            x = idct(x)
            return x
        
        else:
            raise ValueError('Invalid transform type')


class DCT(AbstractLinearFunction):

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
    

class IDCT(AbstractLinearFunction):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = idctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        return x

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = dctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        return x
    

class PatchwiseDCT(AbstractLinearFunction):

    def __init__(self, p1, p2) -> None:
        super().__init__()
        self.patch = Rearrange("b c (h p1) (w p2) -> b h w c p1 p2", p1=p1, p2=p2)
        self.unpatch = Rearrange("b h w c p1 p2 -> b c (h p1) (w p2)", p1=p1, p2=p2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = self.patch(x)
        x = x.detach().cpu().numpy()
        x = dctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        x = self.unpatch(x)
        return x

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = self.patch(x)
        x = x.detach().cpu().numpy()
        x = idctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        x = self.unpatch(x)
        return x
    

class PatchwiseIDCT(AbstractLinearFunction):

    def __init__(self, p1, p2) -> None:
        super().__init__()
        self.patch = Rearrange("b c (h p1) (w p2) -> b h w c p1 p2", p1=p1, p2=p2)
        self.unpatch = Rearrange("b h w c p1 p2 -> b c (h p1) (w p2)", p1=p1, p2=p2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = self.patch(x)
        x = x.detach().cpu().numpy()
        x = idctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        x = self.unpatch(x)
        return x

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = self.patch(x)
        x = x.detach().cpu().numpy()
        x = dctn(x, norm='ortho')
        x = torch.Tensor(x).to(device)
        x = self.unpatch(x)
        return x
    

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

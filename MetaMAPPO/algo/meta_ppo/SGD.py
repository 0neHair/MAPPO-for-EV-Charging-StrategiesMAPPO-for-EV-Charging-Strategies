'''
Author: CQZ
Date: 2025-04-24 14:10:36
Company: SEU
'''
from typing import Tuple
import torch

class SGD(object):
    def __init__(self, module: torch.nn.Module, lr: float):
        self.module = module
        self.param_name = list(dict(module.named_parameters()).keys())
        self.module_name = dict(module.named_modules())
        self.lr = lr

    def step(self, param_grads: Tuple = None):
        if param_grads:
            # use autograd.grad(), called by adapt
            for name, grad in zip(self.param_name, param_grads):
                if "." in name:
                    # policy.0.bias -> module_name: policy.0, param_name: bias
                    module_name, param_name = name.rsplit(".", 1)
                    self.update(self.module_name[module_name], param_name, grad)
                else:
                    self.update(self.module, name, grad)
        else:
            # use loss.backward(), called by meta update
            torch.set_grad_enabled(False)
            for name, param in self.module.named_parameters():
                if param.grad is not None:
                    param.data = param.data - self.lr * param.grad
            torch.set_grad_enabled(True)

    def update(self, module: torch.nn.Module, param_name: str, grad):
        new_param = module._parameters[param_name] - self.lr * grad
        # new_param.retain_grad()
        del module._parameters[param_name]
        setattr(module, param_name, new_param)
        module._parameters[param_name] = new_param

    def zero_grad(self, set_to_none: bool = True):
        for param in self.module.parameters():
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                if param.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()
                
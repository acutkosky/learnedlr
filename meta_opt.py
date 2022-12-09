
import torch
import random
import numpy as np


class CopyWithGrad(torch.autograd.Function):
    '''
    copies data from source tensor into the same storage as dest.
    In terms of future forward passes, this is the same as:
    
    dest.copy_(source)

    However, possibly unlike in-place copying, this will preserve gradient information.
    I think pytorch might actually work fine with .copy_(), but I'm not sure this is
    something that can be relied on in perpetuity.
    '''

    @staticmethod
    def forward(ctx, source, dest):
        dest.copy_(source)

        return dest

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


copy_with_grad = CopyWithGrad.apply


class MetaOpt(torch.optim.Optimizer):


    def __init__(self, params, defaults, depth=0):
        super(MetaOpt, self).__init__(params, defaults)


        self.depth = depth

        # we could do all sorts of fancy manipulation of __setattr__ to help auto-populate these
        # guys, but that seems like a task for another time.

        # submodules should be stored here
        self.named_modules = {}
        self.unnamed_modules = []

        # parameters we want to be meta-learnable are stored here.
        self.named_parameters = {}
        self.unnamed_parameters = []

        # parameters we DON'T want to be meta-learnable are stored here.
        self.named_buffers = {}
        self.unnamed_buffers = []

        # lower order optimizers stored here
        self.lower_opts = []


        self.setup()

    def setup(self):
        # please define all the parameters here. If you define more later things may go wrong!
        pass



    def parameters(self, recurse=False):
        for p in self.unnamed_parameters:
            yield p
        for p in self.named_parameters.values():
            yield p
        
        if recurse:
            for m in self.unnamed_modules:
                for p in m.parameters(recurse):
                    yield p
            for m in self.named_modules.values():
                for p in m.parameters(recurse):
                    yield p

    def modules(self):
        for m in self.unnamed_modules:
            yield m
        for m in self.named_modules.values():
            yield m
                

    def register_parameter(self, parameter, name=None):
        if name is not None:
            self.named_parameters[name]=parameter
        else:
            self.unnamed_parameters.append(parameter)
        return parameter

    def register_buffer(self, buffer, name=None):
        if name is not None:
            self.named_buffers[name]=buffer
        else:
            self.unnamed_buffers.append(buffer)
        return buffer


    def register_module(self, module, name=None):
        if name is not None:
            self.named_modules[name]=module
        else:
            self.unnamed_modules.append(module)
        return module
    

    def zero_grad(self):
        super().zero_grad()
        for p in self.parameters():
            p.grad = None
        for m in self.modules():
            m.zero_grad()
        for m in self.lower_opts:
            m.zero_grad()



    # this is the real step you need to implement.
    # The current parameter values are stored in param.current_value.
    # it is preferred to use these than the actual param as that might mess
    # up gradient comptuations.
    # be careful: we are deliberatedly not wrapped in toch.no_grad,
    # so no in-place operations, with the notable exception of
    # copy_with_grad, which should be used to update the value of each param
    # in-place.
    def _step(self, *args, **kwargs):
        raise NotImplementedError


    def step(self, *args, **kwargs):
        print(f"running step at depth: {self.depth}")
        for group in self.param_groups:
            for param in group['params']:
                print(f"checking param: {param}")
                if param.grad is None:
                    continue
                print(f"prepping param: {param} with grad: {param.grad}")
                param.grad.detach_()

                # save the current value (this seems hacky).. also no sure if the explicit copy here is needed.
                param.current_value = param.detach()

                param.detach_()
                param.requires_grad = True

        

        # capture any return data
        self._step(*args, **kwargs)

        for group in self.param_groups:
            for param in group['params']:
                # the params are not probably intermediate nodes, so we need to retain the grad.
                param.retain_grad()

        for opt in self.lower_opts:
            opt.step()
        

def chain_meta_optimizers(params, to_chain, args=None, kwargs=None):

    if args is None:
        args = [[] for _ in to_chain]
    if kwargs is None:
        kwargs = [{} for _ in to_chain]

    depth = 0

    lower_opt = to_chain[0](params, *args[0], **kwargs[0], depth=depth)

    for opt, args_, kwargs_ in zip(to_chain[1:], args[1:], kwargs[1:]):
        depth += 1
        upper_opt = opt(lower_opt.parameters(), *args_, **kwargs_, depth=depth)
        upper_opt.lower_opts.append(lower_opt)
        lower_opt = upper_opt

    
    return upper_opt


## subclasses:

class SGD(MetaOpt):

    def __init__(self, params, lr, wd=0.0, *args, **kwargs):
        defaults = dict(lr=lr, wd=wd)
        super().__init__(params, defaults, *args, **kwargs)

        self.lr = self.register_parameter(torch.full((1,), lr, requires_grad=True))


    def _step(self, closure=None):

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                print(f"doing sgd on param: {param}, grad: {param.grad}")

                new_param = param - self.lr * param.grad

                copy_with_grad(new_param, param)

def generate():
    a  = torch.tensor([1.0,2.0,3.0], requires_grad=True)
    o = chain_meta_optimizers([a], [SGD, SGD], [[0.1], [0.2]])
    return a, o
def test(a, o):
    b = torch.sum(a*a)
    b.backward()
    o.step()
    o.zero_grad()
    return a,o
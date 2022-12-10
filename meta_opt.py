
import torch
import random
import numpy as np

from inspect import currentframe, getframeinfo

LOGGING_ON = False

def log(msg, *args, **kwargs):
    if not LOGGING_ON:
        return
    
    frameinfo = getframeinfo(currentframe().f_back)
    print(f"[{frameinfo.filename} line: {frameinfo.lineno}] {msg}", *args, **kwargs)


torch.autograd.set_detect_anomaly(True)

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
        self.lower_opt = None
        self.upper_opt = None

        self.__setstate__(self.state)
        self.setup()

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                if '_meta_state' not in state:
                    state['_meta_state'] = {}

                state = state['_meta_state']

                state['current_value'] = None
                state['updated_value'] = None
                state['current_grad'] = None

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
    

    def zero_grad(self, entry=None):
        super().zero_grad()
        for p in self.parameters():
            p.grad = None
        # for m in self.modules():
        #     m.zero_grad()
        if self.lower_opt is not None and entry != 'lower':
            self.lower_opt.zero_grad(entry='upper')
        if self.upper_opt is not None and entry != 'upper':
            self.upper_opt.zero_grad(entry='lower')
        # for m in self.lower_opts:
        #     m.zero_grad()
        # for m in self.upper_opts:
        #     m.zero_grad()



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

    def backward(self):
        for group in self.param_groups:
            for param in group['params']:
                log(f"checking param: {param}")
                if param.grad is None:
                    continue

                state = self.state[param]['_meta_state']

                log(f"prepping param: {param} with grad: {param.grad}")
                # param.grad.detach_()

                if state['updated_value'] is not None:
                    log(f"about to metagrad: updated value: {state['updated_value']} current_grad: {state['current_grad']} current_value: {state['current_value']}")
                    state['updated_value'].backward(param.grad)
                    # param.updated_value.backward(param.grad.detach())

                # state['replicate'].backward(param.grad)

                # save the current value (this seems hacky).. also no sure if the explicit copy here is needed.
                state['current_value'] = param.clone().detach_()
                state['current_grad'] = param.grad.clone().detach_()
                # param.current_value = param.detach().clone()
                # param.current_grad = param.grad.detach().clone()
                log(f"after meta grad: current_grad: {state['current_grad']} current_value: {state['current_value']}")

                param.detach_()
                param.requires_grad = True

    def step(self, entry=None, *args, **kwargs):
        '''
        entry: indicates whether this step was called by the user,
            or is a recursive call. If user, entry=None
            If recursive and was called by the upper_opt, then
            entry='upper', else entry='lower'.
        '''

        log(f"running step at depth: {self.depth}")
        
        if entry is None:
            # find the lowest model and do backprop.
            opt_to_backward = self
            while opt_to_backward.lower_opt is not None:
                opt_to_backward  = opt_to_backward.lower_opt

            while opt_to_backward is not None:
                opt_to_backward.backward()
                opt_to_backward = opt_to_backward.upper_opt


        

        # capture any return data
        self._step(*args, **kwargs)

        # assert len(self.upper_opts) == 0 or len(self.lower_o pts) == 0

        if self.upper_opt is not None and entry != 'upper':
            self.upper_opt.step(entry='lower')
        # for opt in self.upper_opts:
            # opt.step()
        

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # the params are not probably intermediate nodes, so we need to retain the grad.
                log(f"copying param: {param}")
                state = self.state[param]['_meta_state']
                with torch.no_grad():
                    param.copy_(state['updated_value'])#param.updated_value.clone())
                    log(f"param now: {param} current_value: {state['current_value']}")
        

        if self.lower_opt is not None and entry != 'lower':
            self.lower_opt.step(entry='upper')
        # for opt in self.lower_opts:
        #     opt.step()

def chain_meta_optimizers(params, to_chain, args=None, kwargs=None):

    if args is None:
        args = [[] for _ in to_chain]
    if kwargs is None:
        kwargs = [{} for _ in to_chain]

    depth = 0

    first_opt = lower_opt = upper_opt = to_chain[0](params, *args[0], **kwargs[0], depth=depth)

    for opt, args_, kwargs_ in zip(to_chain[1:], args[1:], kwargs[1:]):
        depth += 1
        upper_opt = opt(lower_opt.parameters(), *args_, **kwargs_, depth=depth)
        lower_opt.upper_opt = upper_opt#s.append(upper_opt)
        upper_opt.lower_opt = lower_opt
        lower_opt = upper_opt

    
    return first_opt


def reverse_chain_meta_optimizers(params, to_chain, args=None, kwargs=None):

    if args is None:
        args = [[] for _ in to_chain]
    if kwargs is None:
        kwargs = [{} for _ in to_chain]

    depth = 0

    first_opt = lower_opt = upper_opt = to_chain[0](params, *args[0], **kwargs[0], depth=depth)

    for opt, args_, kwargs_ in zip(to_chain[1:], args[1:], kwargs[1:]):
        depth += 1
        upper_opt = opt(lower_opt.parameters(), *args_, **kwargs_, depth=depth)
        upper_opt.lower_opt = lower_opt#s.append(lower_opt)
        lower_opt.upper_opt = upper_opt
        lower_opt = upper_opt

    
    return upper_opt


## subclasses:


class SGD(MetaOpt):

    def __init__(self, params, lr, wd=0.0, min_bound=None, max_bound=None, *args, **kwargs):
        defaults = dict(lr=lr, wd=wd)
        super().__init__(params, defaults, *args, **kwargs)
        self.min_bound = min_bound
        self.max_bound = max_bound

        self.lr = self.register_parameter(torch.full((1,), lr, requires_grad=True))


    def _step(self, closure=None):

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]['_meta_state']
                log(f"self.lr {self.lr}, current_value: {state['current_value']}, grad: {param.grad}")

                log(f"doing sgd on param: {param} (current value: {state['current_value']}), grad: {param.grad}")

                state['updated_value'] = state['current_value'] - self.lr * state['current_grad']
                if self.min_bound is not None or  self.max_bound is not None:
                    state['updated_value'] = torch.clamp(state['updated_value'], self.min_bound, self.max_bound)

                log(f"new param: {state['updated_value']}")


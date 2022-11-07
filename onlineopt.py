import torch
import random
import numpy as np

SMALL_VALUE=1e-8


class OnlineLearner:
    '''base class for online learners
    '''

    def __init__(self, name, config, outer_learner):
        self.name = name
        self.config = config
        self.outer_learner = outer_learner


    def get_logging_info(self, aux=None):
        '''
        return a dictionary of logging data
        '''
        return {}

    def pre_update(self, info):
        '''
        provide any info needed before update (maybe some form of communication
        between different learners for example, or external parameter values for
        decoupled weight decay).
        '''
        pass

    def update(self, grad):
        '''
        do the update.
        args:
            params_and_grads: list (param, gradient) tuples.
                the param is a pytorch tensor parameter in the model we are training.
                the grad is a gradient (not necessarily the gradient with respect to the param)
                The online learner is maintaining an iterate for each param, so that the 
                param just serves as a key into a dictionary of iterates.
        
        returns nothing (should use get_iterate to get the iterate)
        '''
        raise NotImplementedError

    def post_update(self, info):
        '''
        if something needs to happen after an update, do it here. Could be used 
        for some kind of communication like pre_update.
        '''
        pass

    def get_iterate(self):
        '''
        returns the current iterate.
        '''
        return self.iterate




class ExpMD(OnlineLearner):
    # This algorithm has an optimal dynamic regret guarantee
    # which I cannot say for methods that actually choose a betting fraction intelligently (although it may be true)
    def __init__(self, config, device, outer_learner=None, name='ExpMD', **kwargs):
        super().__init__(name, config, outer_learner)
        self.lrs = torch.tensor([3**k for k in range(-15,5)], device=device)
        self.min_value = config.get('expmd_min', 1e-10)
        self.max_value = config.get('expmd_max', 1e-2)

        self.max_grad = torch.zeros(1, device=device)
        self.max_grad_beta = 0.99

        self.sub_iterates = torch.ones_like(self.lrs) * 1e-8

        self.count = 0.0

        self.iterate = torch.sum(self.sub_iterates)


    def update(self, grad):

        self.count += 1.0
        
        # max_grad = self.max_grad/(1.0 - self.max_grad_beta**self.count)
        # orig_grad = grad

        # grad = torch.clamp(grad, -self.max_grad, self.max_grad)

        # grad = grad/(self.max_grad + SMALL_VALUE)


        # self.max_grad.mul_(self.max_grad_beta)
        # self.max_grad.copy_(torch.max(torch.abs(orig_grad), self.max_grad))


        self.sub_iterates.mul_(torch.exp(-self.lrs*grad - (0.5 + 1.0/self.count)*self.lrs**2 * grad**2))

        self.sub_iterates.clamp_(min=self.min_value, max=self.max_value)

        self.iterate = torch.sum(self.sub_iterates)

    def get_logging_info(self, aux=None):
        return {
            'optimizer/learned_lr': self.get_iterate()
        }



class OGD(OnlineLearner):
    def __init__(self, config, outer_learner, name='OGD', **kwargs):
        super().__init__(name, config, outer_learner)
        self.lr = config.lr
        self.iterate = 0.0
        self.constraint = config.get('constraint', None)

    def update(self, grad):
        self.iterate.add_(grad, alpha=-self.lr)

        if self.contraint is not None:
            self.iterate.mul_(torch.max(1.0, self.contraint/torch.linalg.norm(self.iterate)))

    def get_iterate(self):
        return self.iterate


class AdamW(OnlineLearner):
    def __init__(self, config, outer_learner, param, name='AdamW', **kwargs):
        super().__init__(name, config, outer_learner)

        self.param = param

        self.lr = config.lr
        self.warmed_up_lr = 0.0
        self.iterate = torch.zeros_like(self.param)
        self.M = torch.zeros_like(self.iterate)
        self.V = torch.zeros_like(self.iterate)

        self.epsilon = config.get('epsilon', SMALL_VALUE)

        self.count = 0

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.wd = config.wd

    def warmup_lr(self):
        self.warmed_up_lr = self.lr * min(1, float(self.count) / float(max(1, self.config.warmup_steps)))
        return self.warmed_up_lr



    def update(self, grad):
        self.count += 1
        self.warmup_lr()
        # AdamW update
        self.M.mul_(self.beta1)
        self.M.add_(grad, alpha=1.0-self.beta1)

        self.V.mul_(self.beta2)
        self.V.add_(grad**2, alpha=1.0-self.beta2)

        

        M_hat = self.M/(1.0-self.beta1**self.count)
        V_hat = self.V/(1.0-self.beta2**self.count)

        self.iterate = -self.warmed_up_lr * (M_hat/ (torch.sqrt(V_hat) + self.epsilon) + self.wd * self.param)

    def get_logging_info(self, aux=None):
        return {
            'optimizer/learned_lr': self.warmed_up_lr
        }




class PerTensorScaleAdamW(OnlineLearner):
    def __init__(self, config, outer_learner, param, name='AdamW', **kwargs):
        super().__init__(name, config, outer_learner)

        self.param = param

        self.lr = config.lr
        self.warmed_up_lr = 0.0
        self.iterate = torch.zeros_like(self.param)
        self.M = torch.zeros_like(self.iterate)
        self.V = torch.zeros_like(self.iterate)

        self.scale_learner = ExpMD(config, device=param.device)

        self.epsilon = config.get('epsilon', SMALL_VALUE)

        self.count = 0

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.wd = config.wd


    def warmup_lr(self):
        self.warmed_up_lr = self.lr * min(1, float(self.count) / float(max(1, self.config.warmup_steps)))
        return self.warmed_up_lr


    def compute_meta_gradient(self, grad):
        return torch.sum(grad * self.iterate)/self.scale_learner.get_iterate()


    def update(self, grad):

        meta_gradient = self.compute_meta_gradient(grad)
        self.scale_learner.update(meta_gradient)

        self.count += 1
        self.warmup_lr()
        # AdamW update
        self.M.mul_(self.beta1)
        self.M.add_(grad, alpha=1.0-self.beta1)

        self.V.mul_(self.beta2)
        self.V.add_(grad**2, alpha=1.0-self.beta2)

        

        M_hat = self.M/(1.0-self.beta1**self.count)
        V_hat = self.V/(1.0-self.beta2**self.count)

        self.iterate = -self.scale_learner.get_iterate() * self.warmed_up_lr * (M_hat/ (torch.sqrt(V_hat) + self.epsilon) + self.wd * self.param)

    def get_logging_info(self, aux=None):
        max_lr = max((self.warmed_up_lr * self.scale_learner.get_iterate()).item(), 
            aux.get('optimizer/max_learned_lr', 0.0))
        min_lr = min((self.warmed_up_lr * self.scale_learner.get_iterate()).item(),
            aux.get('optimizer/min_learned_lr', 100.0))
        return {
            'optimizer/max_learned_lr': max_lr,
            'optimizer/min_learned_lr': min_lr
        }


PER_TENSOR_OL_REGISTRY = {
    'adamw': AdamW,
    'ptscaleadamw': PerTensorScaleAdamW,
}

GLOBAL_OL_REGISTRY = {
    'ogd': OGD,
}

class PerTensorRandomOL(torch.optim.Optimizer):

    def __init__(self, params, config, logger, **kwargs):
        
        super().__init__(params, kwargs)

        self.config = config
        self.logger = logger
        self.count = 0

        self.__setstate__(self.state)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['ol'] = PER_TENSOR_OL_REGISTRY[self.config.ol.lower()](self.config, self, param)

                state['iterate'] = param.clone().detach_()
                state['offset'] = torch.zeros_like(param)

        self.state['reward'] = 0.0

    def offset_correlation(self):
        correlations = {}
        total_correlation = 0.0
        for group in self.param_groups:
            for param in group['params']:

                if param.grad is None:
                    continue

                state = self.state[param]

                
                correlation = torch.sum(state['offset'] * param.grad)
                correlations[param] = correlation
                total_correlation += correlation

        return correlations, total_correlation



    
    @torch.no_grad()
    def step(self, closure=None):

        self.count += 1
        logging_info = {}

        if self.config.scale_type == 'random':
            scaling = random.random()
        elif self.config.scale_type == 'half':
            scaling = 0.5
        elif self.config.scale_type == 'one':
            scaling = 1.0
        elif self.config.scale_type ==  'exp':
            scaling = -np.log(1.0-random.random())


        correlations, total_correlation = self.offset_correlation()
        self.state['reward'] -= total_correlation

        logging_info.update({
            'optimizer/total_reward': self.state['reward'],
            'optimizer/current_reward': -total_correlation,
        })

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                state = self.state[param]

                state['ol'].update(param.grad)

                state['offset'].copy_(state['ol'].get_iterate())


                if self.config.scale_type != 'exp':
                    param.copy_(state['iterate'] + scaling*state['offset'])
                    state['iterate'].add_(state['offset'])
                else:
                    param.add_(state['offset'], alpha=scaling)
                    state['iterate'].copy_(param)

                logging_info.update(state['ol'].get_logging_info(logging_info))

        
        self.logger.log(
            logging_info,
            commit=False
        )

        

        








        








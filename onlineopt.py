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

    def update(self, grad, **aux):
        '''
        do the update.
        args:
            grad: gradient information
        
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

        self.sum_mean = config.get('expmd_sum_vs_mean', 'sum')

        self.post_agg_clip = config.get('expmd_post_agg_clip', False)

        self.max_grad = torch.zeros(1, device=device)
        self.max_grad_beta = 0.99

        self.sub_iterates = torch.ones_like(self.lrs) * config.get('expmd_initial_value', 1e-8)

        self.count = 0.0


        # self.iterate = torch.sum(self.sub_iterates)

    def get_iterate(self):
        if self.sum_mean == 'sum':
            iterate = torch.sum(self.sub_iterates)
        else:
            iterate = torch.mean(self.sub_iterates)

        if self.post_agg_clip:
            iterate.clamp_(min=self.min_value, max=self.max_value)

        return iterate



    def update(self, grad):

        self.count += 1.0

        # perform 1d constraint reduction if required:
        if self.post_agg_clip:
            # ignore gradients that push us further outside the bound
            iterate = self.get_iterate()
            if iterate == self.max_value and grad < 0:
                return
            if iterate == self.min_value and grad > 0:
                return
            
        
        # max_grad = self.max_grad/(1.0 - self.max_grad_beta**self.count)
        # orig_grad = grad

        # grad = torch.clamp(grad, -self.max_grad, self.max_grad)

        # grad = grad/(self.max_grad + SMALL_VALUE)


        # self.max_grad.mul_(self.max_grad_beta)
        # self.max_grad.copy_(torch.max(torch.abs(orig_grad), self.max_grad))


        self.sub_iterates.mul_(torch.exp(-self.lrs*grad - (0.5 + 1.0/self.count)*self.lrs**2 * grad**2))

        self.sub_iterates.clamp_(min=self.min_value, max=self.max_value)

        # self.iterate = torch.sum(self.sub_iterates)

    def get_logging_info(self, aux=None):
        return {
            'optimizer/learned_lr': self.get_iterate()
        }




class ExpMD_plusminus(OnlineLearner):
    # This algorithm has an optimal dynamic regret guarantee
    # which I cannot say for methods that actually choose a betting fraction intelligently (although it may be true)
    def __init__(self, config, device, outer_learner=None, name='ExpMD_plusminus', **kwargs):
        super().__init__(name, config, outer_learner)
        self.lrs = torch.tensor([3**k for k in range(-15,5)], device=device)
        self.min_value = config.get('expmd_min', 1e-10)
        self.max_value = config.get('expmd_max', 1e-2)

        self.sum_mean = config.get('expmd_sum_vs_mean', 'sum')

        self.post_agg_clip = config.get('expmd_post_agg_clip', False)

        self.max_grad = torch.zeros(1, device=device)
        self.max_grad_beta = 0.99

        self.sub_iterates_plus = torch.ones_like(self.lrs) * config.get('expmd_initial_value', 1e-8)
        self.sub_iterates_minus = torch.ones_like(self.lrs) * config.get('expmd_initial_value', 1e-8)

        self.count = 0.0


        # self.iterate = torch.sum(self.sub_iterates)

    def get_iterate(self):
        if self.sum_mean == 'sum':
            iterate = torch.sum(self.sub_iterates_plus) - torch.sum(self.sub_iterates_minus)
        else:
            iterate = torch.mean(self.sub_iterates_plus) - torch.mean(self.sub_iterates_minus)

        if self.post_agg_clip:
            iterate.clamp_(min=-1, max=self.max_value)

        return iterate



    def update(self, grad):

        self.count += 1.0

        # perform 1d constraint reduction if required:
        if self.post_agg_clip:
            # ignore gradients that push us further outside the bound
            iterate = self.get_iterate()
            if iterate == self.max_value and grad < 0:
                return
            if iterate == self.min_value and grad > 0:
                return
            
        
        # max_grad = self.max_grad/(1.0 - self.max_grad_beta**self.count)
        # orig_grad = grad

        # grad = torch.clamp(grad, -self.max_grad, self.max_grad)

        # grad = grad/(self.max_grad + SMALL_VALUE)


        # self.max_grad.mul_(self.max_grad_beta)
        # self.max_grad.copy_(torch.max(torch.abs(orig_grad), self.max_grad))


        self.sub_iterates_plus.mul_(torch.exp(-self.lrs*grad - (0.5 + 1.0/self.count)*self.lrs**2 * grad**2))
        self.sub_iterates_minus.mul_(torch.exp(self.lrs*grad - (0.5 + 1.0/self.count)*self.lrs**2 * grad**2))

        self.sub_iterates_plus.clamp_(min=self.min_value, max=self.max_value)
        self.sub_iterates_minus.clamp_(min=self.min_value, max=self.max_value)

        # self.iterate = torch.sum(self.sub_iterates)

    def get_logging_info(self, aux=None):
        param_name = aux['param_name']
        return {
            f'per_tensor_lrs/{param_name}': self.get_iterate()
        }



class ExpMD_twosided(OnlineLearner):
    # This algorithm has an optimal dynamic regret guarantee
    # which I cannot say for methods that actually choose a betting fraction intelligently (although it may be true)
    def __init__(self, config, device, outer_learner=None, name='ExpMD_twosided', **kwargs):
        super().__init__(name, config, outer_learner)
        self.lrs = torch.tensor([3**k for k in range(-15,5)], device=device)
        self.alphas = self.lrs
        self.max_value = config.get('expmd_max', 1)
        self.min_value = config.get('expmd_min', 1)

        self.sum_mean = config.get('expmd_sum_vs_mean', 'sum')

        self.post_agg_clip = config.get('expmd_post_agg_clip', False)

        self.max_grad = torch.zeros(1, device=device)
        self.max_grad_beta = 0.99

        self.sub_iterates = torch.zeros_like(self.lrs)

        self.count = 0.0


        # self.iterate = torch.sum(self.sub_iterates)

    def get_iterate(self):
        if self.sum_mean == 'sum':
            iterate = torch.sum(self.sub_iterates)
        else:
            iterate = torch.mean(self.sub_iterates)

        if self.post_agg_clip:
            iterate.clamp_(min=-self.min_value, max=self.max_value)

        return iterate



    def update(self, grad):

        self.count += 1.0

        # perform 1d constraint reduction if required:
        if self.post_agg_clip:
            # ignore gradients that push us further outside the bound
            iterate = self.get_iterate()
            if iterate == self.max_value and grad < 0:
                return
            if iterate == -self.min_value and grad > 0:
                return
            
        theta = 2 * torch.sign(self.sub_iterates) *torch.log(torch.abs(self.sub_iterates)/self.alphas +1)/self.lrs - grad
        self.sub_iterates.copy_(self.alphas *torch.sign(theta) 
            * (
                torch.exp(0.5 * self.lrs * torch.nn.functional.relu(
                    torch.abs(theta) - 2 * self.lrs * grad**2
                    ))
                    - 1.0
                ))
        
        self.sub_iterates.clamp_(min=-self.min_value, max=self.max_value)

        # self.iterate = torch.sum(self.sub_iterates)

    def get_logging_info(self, aux=None):
        if aux is not None and 'param_name' in aux:
            param_name = aux['param_name']
            return {
                f'per_tensor_lrs/{param_name}': self.get_iterate()
            }
        else:
            return {
                'optimizer/learned_lr': self.get_iterate()
            }






class OGD(OnlineLearner):
    def __init__(self, config, outer_learner, name='OGD', **kwargs):
        super().__init__(name, config, outer_learner)
        self.lr = config.lr
        self.iterate = 0.0
        self.constraint = config.get('ogd_constraint', None)

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

        self.normalize = config.get('adamw_normalize', 'none')

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

        if self.normalize == 'normalize':
            norm_factor = 1.0/(torch.linalg.norm(M_hat/ (torch.sqrt(V_hat) + self.epsilon)) + self.epsilon)
        else:
            norm_factor = 1.0

        self.iterate = -self.warmed_up_lr * (M_hat/ (torch.sqrt(V_hat) + self.epsilon) * norm_factor + self.wd * self.param)

    def get_logging_info(self, aux=None):
        return {
            'optimizer/learned_lr': self.warmed_up_lr
        }




class PerTensorScaleAdamW(OnlineLearner):
    def __init__(self, config, outer_learner, param, name='PerTensorScaleAdamW', **kwargs):
        super().__init__(name, config, outer_learner)

        self.param = param

        self.lr = config.lr
        self.warmed_up_lr = 0.0
        self.iterate = torch.zeros_like(self.param)
        self.M = torch.zeros_like(self.iterate)
        self.V = torch.zeros_like(self.iterate)

        self.scale_learner = ExpMD(config, device=param.device)

        self.normalize = config.get('adamw_normalize', 'none')

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

        if self.normalize == 'normalize':
            norm_factor = 1.0/(torch.linalg.norm(M_hat/ (torch.sqrt(V_hat) + self.epsilon)) + self.epsilon)
        else:
            norm_factor = 1.0

        self.iterate = -self.scale_learner.get_iterate() * self.warmed_up_lr * (M_hat/ (torch.sqrt(V_hat) + self.epsilon) * norm_factor + self.wd * self.param)

    def get_logging_info(self, aux):
        lr = (self.warmed_up_lr * self.scale_learner.get_iterate()).item()
        past_info = aux['log_info']
        param_names = aux['param_names']
        max_lr = max(lr, 
            past_info.get('optimizer/max_learned_lr', 0.0))
        min_lr = min(lr,
            past_info.get('optimizer/min_learned_lr', 100.0))
        param_name = param_names.get(self.param, None)

        log_info = {
            'optimizer/max_learned_lr': max_lr,
            'optimizer/min_learned_lr': min_lr,
        }

        if param_name is not None:
            log_info.update({
                f'per_tensor_lrs/{param_name}': lr,
            })

        return log_info




class GlobalScaleAdamW(OnlineLearner):
    def __init__(self, config, outer_learner, name='GlobalScaleAdamW', **kwargs):
        super().__init__(name, config, outer_learner)


        self.state = {}
        device = None
        for group in outer_learner.param_groups:
            for param in group['params']:
                p_state = {}
                p_state['M'] = torch.zeros_like(param)
                p_state['V'] = torch.zeros_like(param)
                p_state['iterate'] = torch.zeros_like(param)

                self.state[param] = p_state

                device = param.device # this is hacky, idk what to do if params live on multiple devices yet...
        
        self.outer_learner = outer_learner
        
        self.lr = config.lr
        self.warmed_up_lr = 0.0

        self.scale_learner = ExpMD(config, device=device)

        self.normalize = config.get('adamw_normalize', 'none')

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

    def pre_update(self, info):
        meta_gradient = info['total_correlation']/self.scale_learner.get_iterate()
        self.scale_learner.update(meta_gradient)

    def update(self, cors_and_grads):

        self.count += 1
        self.warmup_lr()

        for param in cors_and_grads:
            grad = cors_and_grads[param]['grad']

            state = self.state[param]


            # AdamW update
            state['M'].mul_(self.beta1)
            state['M'].add_(grad, alpha=1.0-self.beta1)

            state['V'].mul_(self.beta2)
            state['V'].add_(grad**2, alpha=1.0-self.beta2)

            

            M_hat = state['M']/(1.0-self.beta1**self.count)
            V_hat = state['V']/(1.0-self.beta2**self.count)

            if self.normalize == 'normalize':
                norm_factor = 1.0/(torch.linalg.norm(M_hat/ (torch.sqrt(V_hat) + self.epsilon)) + self.epsilon)
            else:
                norm_factor = 1.0

            state['iterate'] = -self.scale_learner.get_iterate() * self.warmed_up_lr * (M_hat/ (torch.sqrt(V_hat) + self.epsilon) * norm_factor + self.wd * param)
        
    def get_iterate(self):
        return self.state

    def get_logging_info(self, aux=None):
        return {
            'optimizer/learned_lr': self.scale_learner.get_iterate(),
        }


class PerTensorRandomOL(torch.optim.Optimizer):

    def __init__(self, params, config, logger, named_params=None, **kwargs):
        
        super().__init__(params, kwargs)

        self.config = config
        self.logger = logger
        self.count = 0

        self.param_names = {}

        if named_params is not None:
            for name, param in named_params:
                self.param_names[param] = name

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

    def offset_correlation(self, loss_difference=None):
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

        if loss_difference is not None:
            for param in correlations:
                correlations[param].mul_(-loss_difference/total_correlation)
            total_correlation = -loss_difference

        return correlations, total_correlation



    
    @torch.no_grad()
    def step(self, loss_difference=None, closure=None):

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


        correlations, total_correlation = self.offset_correlation(loss_difference)
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

                logging_info.update(state['ol'].get_logging_info({
                    'log_info': logging_info,
                    'param_names': self.param_names,
                    }))

        
        # self.logger.log(
        #     logging_info,
        #     commit=False
        # )
        return logging_info
        

        





class GlobalRandomOL(torch.optim.Optimizer):

    def __init__(self, params, config, logger, **kwargs):
        
        super().__init__(params, kwargs)

        self.config = config
        self.logger = logger
        self.count = 0

        self.__setstate__(self.state)

        # might be important to initialize this AFTER __setstate__.
        self.ol = GLOBAL_OL_REGISTRY[self.config.ol.lower()](self.config, outer_learner=self)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]

                state['iterate'] = param.clone().detach_()
                state['offset'] = torch.zeros_like(param)

        self.state['reward'] = 0.0

    def offset_cors_and_grads(self, loss_difference=None):
        correlations = {}
        total_correlation = 0.0
        for group in self.param_groups:
            for param in group['params']:

                if param.grad is None:
                    continue

                state = self.state[param]

                
                correlation = torch.sum(state['offset'] * param.grad)
                correlations[param] = {
                    'correlation': correlation,
                    'grad': param.grad
                }

                total_correlation += correlation

        if loss_difference is not None:
            for param in correlations:
                correlations[param]['correlation'].mul_(-loss_difference/total_correlation)
            total_correlation = -loss_difference

        return correlations, total_correlation



    
    @torch.no_grad()
    def step(self, loss_difference=None, closure=None):

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


        cors_and_grads, total_correlation = self.offset_cors_and_grads(loss_difference)
        self.state['reward'] -= total_correlation

        logging_info.update({
            'optimizer/total_reward': self.state['reward'],
            'optimizer/current_reward': -total_correlation,
        })

        self.ol.pre_update({'total_correlation': total_correlation})
        self.ol.update(cors_and_grads)
        new_offsets = self.ol.get_iterate()

        for param in new_offsets:
            state = self.state[param]
            state['offset'].copy_(new_offsets[param]['iterate'])

            if self.config.scale_type != 'exp':
                param.copy_(state['iterate'] + scaling*state['offset'])
                state['iterate'].add_(state['offset'])
            else:
                param.add_(state['offset'], alpha=scaling)
                state['iterate'].copy_(param)  

        logging_info.update(self.ol.get_logging_info(logging_info))          

        # self.logger.log(
        #     logging_info,
        #     commit=False
        # )

        return logging_info

        


class PerTensorRandomResidualOL(torch.optim.Optimizer):

    def __init__(self, params, config, logger, named_params=None, **kwargs):
        
        super().__init__(params, kwargs)

        self.config = config
        self.logger = logger
        self.count = 0

        self.param_names = {}

        if named_params is not None:
            for name, param in named_params:
                self.param_names[param] = name

        self.__setstate__(self.state)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['ol'] = META_OL_REGISTRY[self.config.ol.lower()](self.config, param.device, self)
                state['opt'] = PER_TENSOR_OL_REGISTRY[self.config.opt.lower()](self.config, self, param)

                state['iterate'] = param.clone().detach_()
                state['offset'] = torch.zeros_like(param)

        self.state['reward'] = 0.0

    def offset_correlation(self, loss_difference=None):
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

        if loss_difference is not None:
            for param in correlations:
                correlations[param].mul_(-loss_difference/total_correlation)
            total_correlation = -loss_difference

        return correlations, total_correlation



    
    @torch.no_grad()
    def step(self, loss_difference=None, closure=None):

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


        correlations, total_correlation = self.offset_correlation(loss_difference)
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


                residual_grad = correlations[param]
                state['ol'].update(residual_grad)
                state['opt'].update(param.grad)

                state['offset'].copy_(state['opt'].get_iterate())


                param.add_((scaling*state['ol'].get_iterate() + 1.0)*state['offset'])


                logging_info.update(state['opt'].get_logging_info())

                logging_info.update(state['ol'].get_logging_info({
                    'log_info': logging_info,
                    'param_name': self.param_names[param],
                    }))

        
        # self.logger.log(
        #     logging_info,
        #     commit=False
        # )
        return logging_info
        

        
        




PER_TENSOR_OL_REGISTRY = {
    'adamw': AdamW,
    'ptscaleadamw': PerTensorScaleAdamW,
}

GLOBAL_OL_REGISTRY = {
    'gscaleadamw': GlobalScaleAdamW,
}

META_OL_REGISTRY = {
    'expmd_twosided': ExpMD_twosided,
}





        








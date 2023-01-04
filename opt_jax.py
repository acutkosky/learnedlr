
import jax
from jax import numpy as jnp
import numpy as np
import functools

from jax.tree_util import tree_map, tree_reduce

def broadcast(to_broadcast, broadcast_to):
    return tree_map(lambda b, postfix: tree_map(lambda x: b, postfix),
        to_broadcast,
        broadcast_to)
    
def tree_dot(a, b):
    return tree_reduce(
        lambda s, x: s + x,
        tree_map(
            lambda x, y: jnp.sum(x * y),
            a,
            b
        )
    )

def adamw_init(params, lr=1e-3, beta1=0.9, beta2=0.99, wd=0.0, epsilon=1e-8):
    adamw_state = {
        'lr': jnp.array(lr),
        'beta1': jnp.array(beta1),
        'beta2': jnp.array(beta2),
        'wd': jnp.array(wd),
        'epsilon': jnp.array(epsilon),
        'count': jnp.array(0),
        'm': tree_map(lambda x: jnp.zeros_like(x), params),
        'v': tree_map(lambda x: jnp.zeros_like(x), params),
    }

    return adamw_state



def adamw(loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):
    '''
        computes and optimizer update step.
            rng: jax rng
            loss_and_grad_fn: a function that will return a tuple loss, grad.
                this function shouldtake as input a single jax tree argument.
            model_and_example_state: input to loss_and_grad_fn (probably a
                combination of the model state and the example/minibatch)
                This should be a dict for which one key is 'model_state'.

                model_and_example_state['model_state'] should be a flax
                model_state object, so it should have a key 'params' which
                contains model params and a key 'constants' which contains constants
                and nothing else (probably this could be generalized a bit).

                The grad value returned by loss_and_grad_fn should have the 
                same pytree shape as params.

            opt_state: state of optimizer
            scale: scaling value (e.g. a learning rate).
        
        returns:
            rng, value, grad, model_state, opt_state, log_dict

            rng: next value for jax rng
            value: value returned by loss_and_grad_fn
            grad: gradient returned by loss_and_grad_fn
            model_state: next value of the model_state
            opt_state: next state of optimizer
            log_dict: dictionary of keys to be logged in wandb (can be set to
                None to log nothing)
    '''

    value, grad = loss_and_grad_fn(model_and_example_state)

    model_state = model_and_example_state['model_state']

    params = model_state['params']
    constants = model_state['constants']
    # other_values = {key: value for key, value in model_state.iter_items() if key != params}

    m_cur = opt_state['m']
    v_cur = opt_state['v']
    lr = opt_state['lr']
    beta1 = opt_state['beta1']
    beta2 = opt_state['beta2']
    wd = opt_state['wd']
    epsilon = opt_state['epsilon']
    count = opt_state['count']
    

    m_next = tree_map(lambda old_m, g: old_m * beta1 + g * (1.0 - beta1), m_cur, grad)

    v_next = tree_map(lambda old_v, g: old_v * beta2 + (g**2) * (1.0 - beta2), v_cur, grad)
    
    count_next = count + 1


    opt_state_next = {
        'lr': lr,
        'm': m_next,
        'v': v_next,
        'beta1': beta1,
        'beta2': beta2,
        'wd': wd,
        'epsilon': epsilon,
        'count': count_next,
    }

    m_hat = tree_map(lambda m: m / (1.0 - beta1**count_next), m_next)
    v_hat = tree_map(lambda v: v / (1.0 - beta2**count_next), v_next)

    param_next = tree_map(
        lambda p, m, v: p - scale * (lr * m/(epsilon+jnp.sqrt(v)) + wd * p),
        params,
        m_hat,
        v_hat)

    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {}
    else:
        log_dict = None

    return rng, value, grad, model_state_next, opt_state_next, log_dict


def adamw_learned_lr_init(params, ol_init, lr=1e-3, beta1=0.9, beta2=0.99, wd=0.0, epsilon=1e-8, lower_bound=1e-8, upper_bound=10, *args, **kwargs):
    state = {
        'lr': jnp.array(lr),
        'beta1': jnp.array(beta1),
        'beta2': jnp.array(beta2),
        'wd': jnp.array(wd),
        'epsilon': jnp.array(epsilon),
        'count': jnp.array(0),
        'm': tree_map(lambda x: jnp.zeros_like(x), params),
        'v': tree_map(lambda x: jnp.zeros_like(x), params),
        'prev_update': tree_map(lambda x: jnp.zeros_like(x), params),
        'prev_lr_offset': jnp.zeros(1),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'ol_state': ol_init(jnp.zeros(1), *args, **kwargs),
    }
    return state




def adamw_learned_lr_update(ol_update, loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):
    '''
        computes and optimizer update step.
            rng: jax rng
            loss_and_grad_fn: a function that will return a tuple loss, grad.
                this function shouldtake as input a single jax tree argument.
            model_and_example_state: input to loss_and_grad_fn (probably a
                combination of the model state and the example/minibatch)
                This should be a dict for which one key is 'model_state'.

                model_and_example_state['model_state'] should be a flax
                model_state object, so it should have a key 'params' which
                contains model params and a key 'constants' which contains constants
                and nothing else (probably this could be generalized a bit).

                The grad value returned by loss_and_grad_fn should have the 
                same pytree shape as params.

            opt_state: state of optimizer
            scale: scaling value (e.g. a learning rate).
        
        returns:
            rng, value, grad, model_state, opt_state, log_dict

            rng: next value for jax rng
            value: value returned by loss_and_grad_fn
            grad: gradient returned by loss_and_grad_fn
            model_state: next value of the model_state
            opt_state: next state of optimizer
            log_dict: dictionary of keys to be logged in wandb (can be set to
                None to log nothing)
    '''

    value, grad = loss_and_grad_fn(model_and_example_state)

    model_state = model_and_example_state['model_state']

    params = model_state['params']
    constants = model_state['constants']
    # other_values = {key: value for key, value in model_state.iter_items() if key != params}

    m_cur = opt_state['m']
    v_cur = opt_state['v']
    lr = opt_state['lr']
    beta1 = opt_state['beta1']
    beta2 = opt_state['beta2']
    wd = opt_state['wd']
    epsilon = opt_state['epsilon']
    count = opt_state['count']
    prev_update = opt_state['prev_update']
    prev_lr_offset = opt_state['prev_lr_offset']
    ol_state = opt_state['ol_state']
    lower_bound = opt_state['lower_bound']
    upper_bound = opt_state['upper_bound']


    ol_prediction = ol_state['prediction']


    scale_grad = tree_dot(prev_update, grad)


    cur_rng, rng  = jax.random.split(rng)

    rand_scaling = jax.random.uniform(cur_rng)


    # apply constraint set reduction for 1-D intervals
    scale_grad = jnp.where(
        -(ol_prediction - prev_lr_offset) * jnp.sign(scale_grad) > 1e-8, # should be == 0 when no clipping happens, but idk about floating point stuff.
        jnp.zeros_like(scale_grad),
        scale_grad)

    ol_state, ol_logs = ol_update(scale_grad, ol_state)



    offset = jnp.clip(ol_state['prediction'], a_min=lower_bound*scale-scale, a_max=upper_bound*scale-scale)
    

    m_next = tree_map(lambda old_m, g: old_m * beta1 + g * (1.0 - beta1), m_cur, grad)

    v_next = tree_map(lambda old_v, g: old_v * beta2 + (g**2) * (1.0 - beta2), v_cur, grad)
    
    count_next = count + 1

    m_hat = tree_map(lambda m: m / (1.0 - beta1**count_next), m_next)
    v_hat = tree_map(lambda v: v / (1.0 - beta2**count_next), v_next)

    update = tree_map(
        lambda m, v, p: -lr * m/(epsilon+jnp.sqrt(v)) - wd * p,
        m_hat,
        v_hat,
        params,
    )

    param_next = tree_map(
        lambda p, u: p + (scale + rand_scaling * offset) * u,
        params,
        update)



    opt_state_next = {
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'wd': wd,
        'epsilon': epsilon,
        'count': count_next,
        'm': m_next,
        'v': v_next,
        'prev_update': update,
        'prev_lr_offset': offset,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'ol_state': ol_state,
    }

    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {
            'learned_lr': scale + offset,
        }
        log_dict.update(ol_logs)
    else:
        log_dict = None

    return rng, value, grad, model_state_next, opt_state_next, log_dict










def adamw_learned_per_layer__lr_init(params, ol_init, lr=1e-3, beta1=0.9, beta2=0.99, wd=0.0, epsilon=1e-8, lower_bound=1e-8, upper_bound=10, *args, **kwargs):
    state = {
        'lr': jnp.array(lr),
        'beta1': jnp.array(beta1),
        'beta2': jnp.array(beta2),
        'wd': jnp.array(wd),
        'epsilon': jnp.array(epsilon),
        'count': jnp.array(0),
        'm': tree_map(lambda x: jnp.zeros_like(x), params),
        'v': tree_map(lambda x: jnp.zeros_like(x), params),
        'prev_update': tree_map(lambda x: jnp.zeros_like(x), params),
        'prev_lr_offset': jnp.zeros(1),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'ol_state': ol_init(tree_map(lambda x: jnp.zeros(1), params), *args, **kwargs),
    }
    return state




def adamw_learned_per_layer_lr_update(ol_update, loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):
    '''
        computes and optimizer update step.
            ol_update: this is the online learner updater.
                currently should just be cb_update.
            rng: jax rng
            loss_and_grad_fn: a function that will return a tuple loss, grad.
                this function should take as input a single jax tree argument.
            model_and_example_state: input to loss_and_grad_fn (probably a
                combination of the model state and the example/minibatch)
                This should be a dict for which one key is 'model_state'.

                model_and_example_state['model_state'] should be a flax
                model_state object, so it should have two keys, 'params' and 'constants'.
                'params' should contain model params and 'constants' should
                contains constants (probably this could be generalized a bit in future).

                The grad value returned by loss_and_grad_fn should have the 
                same pytree shape as params.

            opt_state: state of optimizer
            scale: scaling value (e.g. a learning rate).

            do_logging: boolean for whether to output logging info.
                WARNING: if you jit this function, you may want to set
                do_logging as a static argument because it is used in a plain old
                python "if" statement (although it may be ok if you only ever use
                do_logging=True or do_logging=False and never switch).
        
        returns:
            rng, value, grad, model_state, opt_state, log_dict

            rng: next value for jax rng
            value: value returned by loss_and_grad_fn
            grad: gradient returned by loss_and_grad_fn
            model_state: next value of the model_state
            opt_state: next state of optimizer
            log_dict: dictionary of keys to be logged in wandb (will be None if do_logging is False)
    '''

    value, grad = loss_and_grad_fn(model_and_example_state)

    model_state = model_and_example_state['model_state']

    params = model_state['params']
    constants = model_state['constants']
    # other_values = {key: value for key, value in model_state.iter_items() if key != params}

    m = opt_state['m']
    v = opt_state['v']
    lr = opt_state['lr']
    beta1 = opt_state['beta1']
    beta2 = opt_state['beta2']
    wd = opt_state['wd']
    epsilon = opt_state['epsilon']
    count = opt_state['count']
    prev_update = opt_state['prev_update']
    prev_lr_offset = opt_state['prev_lr_offset']
    ol_state = opt_state['ol_state']
    lower_bound = opt_state['lower_bound']
    upper_bound = opt_state['upper_bound']


    ol_prediction = ol_state['prediction']


    scale_grad = tree_map(
        lambda u, g: jnp.sum(u*g),
        prev_update, grad
        )
        


    cur_rng, rng  = jax.random.split(rng)

    rand_scaling = jax.random.uniform(cur_rng)


    # apply constraint set reduction for 1-D intervals
    scale_grad = tree_map(
        lambda ol, prev, sg: jnp.where(
            -(ol - prev) * jnp.sign(sg) > 1e-8, # should be == 0 when no clipping happens, but idk about floating point stuff.
            jnp.zeros_like(sg),
            sg),
        ol_prediction,
        prev_lr_offset,
        scale_grad)
        

    ol_state, ol_logs = ol_update(scale_grad, ol_state)



    offset = tree_map(
        lambda pred: jnp.clip(pred, a_min=lower_bound*scale-scale, a_max=upper_bound*scale-scale),
        ol_state['prediction'])
    

    m_next = tree_map(lambda old_m, g: old_m * beta1 + g * (1.0 - beta2), m, grad)

    v_next = tree_map(lambda old_v, g: old_v * beta2 + (g**2) * (1.0 - beta2), v, grad)
    
    count_next = count + 1

    m_hat = tree_map(lambda m: m / (1.0 - beta1**count_next), m_next)
    v_hat = tree_map(lambda v: v / (1.0 - beta2**count_next), v_next)

    update = tree_map(
        lambda m, v, p: -lr * m/(epsilon+jnp.sqrt(v)) - wd * p,
        m_hat,
        v_hat,
        params,
    )

    param_next = tree_map(
        lambda p, u, o: p + (scale + rand_scaling * o) * u,
        params,
        update,
        offset)



    opt_state_next = {
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'wd': wd,
        'epsilon': epsilon,
        'count': count_next,
        'm': m_next,
        'v': v_next,
        'prev_update': update,
        'prev_lr_offset': offset,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'ol_state': ol_state,
    }

    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {
            'learned_lr': scale + offset,
        }
        log_dict.update(ol_logs)
    else:
        log_dict = None

    return rng, value, grad, model_state_next, opt_state_next, log_dict









def exp_md_pos_init(params, grid_size=10, min_eta=1e-8, max_eta=1.0):
    thetas = [tree_map(lambda x: jnp.zeros_like(x), params) for _ in range(grid_size)]
    iterates = [tree_map(lambda x: jnp.zeros_like(x), params) for _ in range(grid_size)]

    etas = [jnp.array(min_eta * (max_eta/min_eta)**(k/grid_size)) for k in range(grid_size)]


    # etas = np.logspace(min_eta, max_eta, grid_size)

    alphas = [tree_map(lambda x: jnp.ones_like(x), params) for _ in range(grid_size)]

    opt_state = {
        'thetas': thetas,
        'etas': etas,
        'iterates': iterates,
        'alphas': alphas,
    }

    return opt_state


def exp_md_pos_update(grad, opt_state, max_bound=None):

    thetas = opt_state['thetas']
    iterates = opt_state['iterates']
    etas = opt_state['etas']
    alphas =  opt_state['alphas']

    max_bound = broadcast(max_bound, grad)

    if max_bound is not None:
        grad = [
            tree_map(
                lambda g, i, b: g*jnp.logical_or(i < b, g > 0),
                grad, iterate, max_bound)
                for iterate in iterates
        ]
    else:
        grad = [grad for iterate in iterates]

    thetas_next = [
        tree_map(lambda t, g: t - g - 0.5 * g**2 * eta, theta, g)
        for theta, g, eta in zip(thetas, grad, etas)
    ]

    alphas_next = tree_map(lambda a,g: a+ g**2, alphas, grad)

    # psi(x) = 1/eta * (x * log(x eta/a) - x)
    # psi'(x) = 1/eta * log(x eta/a)
    # psi'(t)^{-1} = eta/a * exp(eta * t)
    iterates_next = tree_map(
        lambda eta, theta, alpha: tree_map(lambda t, a : eta/a * jnp.exp(t * eta), theta, alpha),
        etas, thetas_next, alphas_next)



    opt_state_next = {
        'thetas': thetas_next,
        'etas': etas,
        'iterates': iterates_next,
        'alphas': alphas_next,
    }

    summed_iterate = functools.reduce(lambda x, y: tree_map(lambda a,b: a+b, x,y), iterates_next)

    if max_bound is not None:
        summed_iterate = tree_map(lambda x, b: jnp.clip(x, a_max=b), summed_iterate, max_bound)

            # lambda b, i: tree_map(
            #     lambda x: jnp.clip(x, a_max=b), i
            # ), max_bound, summed_iterate)

    return summed_iterate, opt_state_next

def exp_md_pm_init(params, grid_size=10, min_eta=1e-8, max_eta=1.0):
    opt_state = {
        'pos': exp_md_pos_init(params, grid_size ,min_eta, max_eta),
        'neg': exp_md_pos_init(params, grid_size ,min_eta, max_eta),
    }


    return opt_state

def exp_md_pm_update(grad, opt_state, max_bound=None, min_bound=None):
    opt_state_pos = opt_state['pos']
    opt_state_neg = opt_state['neg']

    neg_grad = tree_map(lambda x: -x, grad)
    if min_bound is not None:
        min_bound = tree_map(lambda x: -x, min_bound)

    
    iterate_pos, opt_state_pos_next = exp_md_pos_update(grad, opt_state_pos, max_bound)
    iterate_neg, opt_state_neg_next = exp_md_pos_update(neg_grad, opt_state_neg, min_bound)

    param_next = tree_map(lambda p, n: p-n, iterate_pos, iterate_neg)

    opt_state_next = {
        'pos': opt_state_pos_next,
        'neg': opt_state_neg_next
    }

    return param_next, opt_state_next


def ogd_init(params, lr):
    return {
        'value': tree_map(lambda x : jnp.zeros_like(x), params),
        'lr': lr
    }

def ogd_update(grad, opt_state, max_bound=None, min_bound=None):
    value = opt_state['value']
    lr = opt_state['lr']

    if min_bound is not None:
        min_bound = broadcast(min_bound, value)

    if max_bound is not None:
        max_bound = broadcast(max_bound, value)
        
    
    
    value_next = tree_map(lambda v, g: v - g*lr, value, grad)

    value_next = tree_map(lambda x, b1, b2: jnp.clip(x, a_max=b1, a_min=b2), value_next, max_bound, min_bound)

    opt_state_next = {
        'value': value_next,
        'lr': lr
    }

    return value_next, opt_state_next

def cb_init(params, eps=1.0, eta=2.0/(2-np.log(3)), decay=1.0):
    state = {}
    state['wealth'] = tree_map(lambda x: jnp.full_like(x, fill_value=eps), params)
    state['bet_fractions'] = tree_map(lambda x: jnp.zeros_like(x), params)
    state['bet_grad_squared_sum'] = tree_map(lambda x: jnp.full_like(x, fill_value=1e-8), params)
    state['max_grads'] = tree_map(lambda x: jnp.full_like(x, fill_value=1e-8), params)
    state['prediction'] = tree_map(lambda x: jnp.zeros_like(x), params)
    state['grad_sum'] = tree_map(lambda x: jnp.zeros_like(x), params)
    state['eta'] = eta
    state['eps'] = eps
    state['decay'] = decay


    return state


def cb_reset(state):
    state['wealth'] = tree_map(lambda x: jnp.full_like(x, fill_value=state['eps']), state['wealth'])
    state['bet_fractions'] = tree_map(lambda x: jnp.zeros_like(x), state['bet_fractions'])
    state['bet_grad_squared_sum'] = tree_map(lambda x: jnp.full_like(x, fill_value=1e-8), state['bet_grad_squared_sum'])
    state['max_grads'] = tree_map(lambda x: jnp.full_like(x, fill_value=1e-8), state['max_grads'])
    state['prediction'] = tree_map(lambda x: jnp.zeros_like(x), state['prediction'])
    return state    

def cb_update(grad, opt_state, do_logging=False):
    
    wealth = opt_state['wealth']
    bet_fractions = opt_state['bet_fractions']
    bet_grad_squared_sum = opt_state['bet_grad_squared_sum']
    max_grads = opt_state['max_grads']
    eta = opt_state['eta']
    eps = opt_state['eps']
    decay = opt_state['decay']
    grad_sum = opt_state['grad_sum']




    max_grads_next = tree_map(lambda m, g: jnp.maximum(m * decay, jnp.abs(g)), max_grads, grad)

    grad = tree_map(lambda g, m: jnp.clip(g, a_min=-m, a_max=m), grad, max_grads)


    grad_sum_next = tree_map(lambda s, g: s * decay + g, grad_sum, grad)

    bet_grad = tree_map(
        lambda g, b: g/(1 - decay * g* b),
        grad,
        bet_fractions
    )

    bet_grad_squared_sum_next = tree_map(lambda x, z: x*decay**2 + z**2, bet_grad_squared_sum,  bet_grad)
    wealth_next = tree_map(lambda r, g, b: r * (decay - b * g), wealth, bet_fractions, grad)
    
    
    bet_fractions_next = tree_map(
        lambda b, z, s, m: jnp.clip(
            b - z * eta/s,
            a_min=-0.5/m, a_max=0.5/m),
        bet_fractions,
        bet_grad,
        bet_grad_squared_sum_next,
        max_grads_next)

    param_next = tree_map(lambda r, b: r * b, wealth_next, bet_fractions_next)

    opt_state_next = {
        'wealth': wealth_next,
        'bet_fractions': bet_fractions_next,
        'bet_grad_squared_sum': bet_grad_squared_sum_next,
        'max_grads': max_grads_next,
        'eta': eta,
        'eps': eps,
        'decay': decay,
        'prediction': param_next,
        'grad_sum': grad_sum_next,
    }

    return opt_state_next, {
        'grad_sum': tree_map(lambda x: jnp.average(x), grad_sum_next),
        'wealth': tree_map(lambda x: jnp.average(x), wealth_next),
        'bet_fractions': tree_map(lambda x: jnp.average(x), bet_fractions_next),
        'max_grad': tree_map(lambda x: jnp.average(x), max_grads_next),
        'unconstrained_lr': tree_map(lambda x: jnp.average(x), param_next),
        }



    

def OL_momentum_init(params, ol_init, *args, **kwargs):
    state = {
        'ol_state': ol_init(params, *args, **kwargs),
        'true_params': params,
        'last_offset': tree_map(lambda x: jnp.zeros_like(x), params),
        'total_reward': 0.0,
        'epoch_reward': 0.0,
        # 'epoch_count': 0,
        'iteration_count':  0,
        # 'reset_threshold': reset_threshold,
        # 'reset_fn': jax.jit(ol_reset),
    }

    return state

def OL_momentum_update(ol_update, loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):

    value, grad = loss_and_grad_fn(model_and_example_state)
    model_state = model_and_example_state['model_state']

    cur_rng, rng = jax.random.split(rng)

    rand_scaling = jax.random.uniform(cur_rng)
 
    params = model_state['params']
    constants = model_state['constants']

    ol_state = opt_state['ol_state']
    true_params = opt_state['true_params']
    total_reward = opt_state['total_reward']
    epoch_reward = opt_state['epoch_reward']
    last_offset = opt_state['last_offset']
    # epoch_count = opt_state['epoch_count']
    iteration_count = opt_state['iteration_count']
    # reset_fn = opt_state['reset_fn']
    # reset_threshold = opt_state['reset_threshold']

    iteration_count_next = iteration_count + 1

    ol_prediction = ol_state['prediction']

    reward_increment = tree_reduce(
        lambda s, x: s+x,
        tree_map(
            lambda o, g: jnp.sum(o*g),
            last_offset,
            grad
        )
    )

    total_reward_next = total_reward - reward_increment
    epoch_reward_next = epoch_reward - reward_increment

    

    # apply constraint set reduction for 1-D intervals
    grad = tree_map(
        lambda o, g: jnp.where(-o * jnp.sign(g) >= scale, jnp.zeros_like(g), g),
        ol_prediction,
        grad)

    ol_state_next, ol_logs = ol_update(grad, ol_state, do_logging)

    offset = tree_map(
        lambda p: jnp.clip(p, a_min=-scale, a_max=scale),
        ol_state_next['prediction']
    )

    params_next = tree_map(lambda p, o: p+ rand_scaling * o, true_params, offset)


    model_state_next = {
        'constants': constants,
        'params': params_next
    }

    opt_state_next = {
        'ol_state': ol_state_next,
        'true_params': tree_map(lambda p, o: p+ o, true_params, offset),
        'total_reward': total_reward_next,
        'iteration_count': iteration_count_next,
        # 'epoch_count': epoch_count_next,
        # 'reset_fn': reset_fn,
        # 'reset_threshold': reset_threshold,
        'last_offset': offset,
    }


    if do_logging:
        log_dict = {
            'total_reward': total_reward_next
        }
        log_dict.update(ol_logs)
    else:
        log_dict = None


    return rng, value, grad, model_state_next, opt_state_next, log_dict

    
   





def OL_momentum_ogd_init(params, ol_lr):
    state = {
        'ogd_state': ogd_init(params, lr=ol_lr),
        'true_params': params
    }
    return state

def OL_momentum_ogd_update(loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):

    value, grad = loss_and_grad_fn(model_and_example_state)
    model_state = model_and_example_state['model_state']

    cur_rng, rng = jax.random.split(rng)

    scaling = jax.random.uniform(cur_rng)

    params = model_state['params']
    constants = model_state['constants']

    ogd_state = opt_state['ogd_state']
    true_params = opt_state['true_params']

    offset, ogd_state_next = ogd_update(grad, ogd_state, min_bound=-scale, max_bound=scale)

    params_next = tree_map(lambda p, o: p+ scaling * o, true_params, offset)


    model_state_next = {
        'constants': constants,
        'params': params_next
    }

    opt_state_next = {
        'ogd_state': ogd_state_next,
        'true_params': tree_map(lambda p, o: p+ o, true_params, offset)
    }


    if do_logging:
        log_dict = {}
    else:
        log_dict = None


    return rng, value, grad, model_state_next, opt_state_next, log_dict

    


def OL_momentum_expmd_init(params, grid_size=10, min_eta=1e-8, max_eta=1.0):
    ol_state = exp_md_pm_init(params, grid_size, min_eta, max_eta)
    return ol_state

def OL_momentum_expmd_update(loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):

    value, grad = loss_and_grad_fn(model_and_example_state)
    model_state = model_and_example_state['model_state']

    params = model_state['params']
    constants = model_state['constants']

    offset, opt_state_next = exp_md_pm_update(grad, opt_state, min_bound=-scale, max_bound=scale)

    param_next = tree_map(lambda p, o: p+o, params, offset)


    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {}
    else:
        log_dict = None


    return rng, value, grad, model_state_next, opt_state_next, log_dict

    
rng = jax.random.PRNGKey(0)
loss_fn = lambda x:  jnp.sum(jnp.zeros(2) * x['r'])

val_and_grad = jax.value_and_grad(loss_fn)

def loss_and_grad_fn(model_and_example_state):
    params = model_and_example_state['model_state']['params']
    return val_and_grad(params)

OL_momentum_expmd_init

model_state = {
    'constants': {'c': jnp.ones(2)},
    'params': {'r': jnp.array([2.0,3.0])}
}

model_and_example_state = {
    'model_state': model_state
}



opt_state = OL_momentum_init(model_state['params'], ol_init=cb_init, decay=0.99)

OL_momentum_update_jit = jax.jit(functools.partial(OL_momentum_update, cb_update, loss_and_grad_fn), static_argnames='do_logging')

# OL_momentum_update_jit =  functools.partial(OL_momentum_update, cb_update, loss_and_grad_fn)


for _ in range(1):
    rng, value, grad, model_state, opt_state, log_dict = OL_momentum_update_jit(rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False)
    model_and_example_state = {
        'model_state': model_state
    }


print(model_state)
print(opt_state)


# opt_state = OL_momentum_expmd_init(model_state['params'], grid_size=2)



# rng, value, grad, model_state, opt_state, log_dict = OL_momentum_expmd_update(rng, loss_and_grad_fn, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False)
# model_and_example_state = {
#     'model_state': model_state
# }
# rng, value, grad, model_state, opt_state, log_dict = OL_momentum_expmd_update(rng, loss_and_grad_fn, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False)
# model_and_example_state = {
#     'model_state': model_state
# }
# rng, value, grad, model_state, opt_state, log_dict = OL_momentum_expmd_update(rng, loss_and_grad_fn, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False)
# model_and_example_state = {
#     'model_state': model_state
# }
# rng, value, grad, model_state, opt_state, log_dict = OL_momentum_expmd_update(rng, loss_and_grad_fn, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False)
# model_and_example_state = {
#     'model_state': model_state
# }


# print(model_state)
# print(opt_state)


# params = {'r': jnp.ones(3)}#,# [jnp.zeros(4), jnp.ones(2)]]

# grads = {'r': jnp.zeros(3)}#, [jnp.zeros(4), jnp.zeros(2)]]

# grads_one = {'r': jnp.ones(3)}#, [jnp.ones(4), jnp.ones(2)]]

# opt_state = exp_md_pm_init(params, grid_size=2)

# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# param_next, opt_state = exp_md_pm_update(grads_one, opt_state, min_bound=-1, max_bound=1)
# print(opt_state)

# print(f"\n\nparam_next: {param_next}")
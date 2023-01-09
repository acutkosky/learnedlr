
import jax
from jax import numpy as jnp
import numpy as np
import functools

import optax

from jax.tree_util import tree_map, tree_reduce

def tree_average(tree):
    return tree_map(lambda x: jnp.average(x), tree)

def zeros_like(tree):
    return tree_map(lambda x: jnp.zeros_like(x), tree)

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

def adamw_init(params, beta1=0.9, beta2=0.99, wd=0.0, epsilon=1e-8):
    adamw_state = {
        'beta1': jnp.array(beta1),
        'beta2': jnp.array(beta2),
        'wd': jnp.array(wd),
        'epsilon': jnp.array(epsilon),
        'count': jnp.array(0),
        'm': zeros_like(params),
        'v': zeros_like(params),
    }

    return adamw_state



def adamw(loss_and_grad_fn, rng, model_and_example_state, opt_state, lr=jnp.array(1.0), do_logging=False):
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
            lr: scaling value (e.g. a learning rate).

            do_logging: flag for whether to output logging info
        
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
    beta1 = opt_state['beta1']
    beta2 = opt_state['beta2']
    wd = opt_state['wd']
    epsilon = opt_state['epsilon']
    count = opt_state['count']
    

    m_next = tree_map(lambda old_m, g: old_m * beta1 + g * (1.0 - beta1), m_cur, grad)

    v_next = tree_map(lambda old_v, g: old_v * beta2 + (g**2) * (1.0 - beta2), v_cur, grad)
    
    count_next = count + 1


    opt_state_next = {
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
        lambda p, m, v: p - lr * ( m/(epsilon+jnp.sqrt(v)) + wd * p),
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

def optax_learned_lr_init(
    params,
    optax_optimizer,
    optax_args,
    optax_kwargs,
    ol_init,
    ol_args,
    ol_kwargs,
    ol_update_fn,
    lower_bound=1e-8,
    upper_bound=10,
    clip=1.0,
    clip_meta_grad=1.0,
    steps_per_ol_update=100,
    multiply=False,
    additive_bounds=False,
    use_loss_diff=False,
    use_rand_scaling=True,
    per_variable_lr=False,
    **_kwargs):
    '''
    learns a (residual) learning rate for adamw, as implemented by optax.

    arguments:
        params: model parameters (pytree)

        optax_optimizer: optimizer class from optax to use as base update generator.
        optax_args: argument list for optax optimizer
        optax_kwargs: keyword argument dict for optax optimizer

        ol_init: function that returns the initial state for an online learner.
        ol_args: argument list for ol_init
        ol_kwargs: keyword argument dict for ol_init.

        ol_update_fn: update function for the online learner.

        lower_bound: the learned learning rate, base_lr + residual, is not allowed to be smaller than lower_bound * base_lr
        upper_bound: the learned learning rate, base_lr + residual, is not allowed to be bigger than upper_bound * base_lr
        clip: clip each gradient to this norm.
        clip_meta_grad: clip the "meta gradient" supplied to the online learner to this value.
        steps_per_ol_update: update the online learner this every steps_per_ol_update steps.

        multiply: if true, learn a multiplicative residual: lr = base_lr * exp(residual). If false, do an additive lr = base_lr + residual
        additive_bounds: if true, change the behavior of lower and upper bounds to be base_lr - lower_bound and base_lr + upper_bound.
                         This option is only supported in combination with multiply=False

        use_loss_diff: if true, then the optimizer uses direct calculuation of difference
            of losses rather than <gradient, update> inner product for the meta gradient signal.
            This works only if we are only learning a scalar right learning rate.
        use_rand_scaling: if true, then we use the randomized uniform scaling in the update function.
        per_variable_lr: if true, learn a separate learning rate for each variable.

    returns:
        state: pytree optimizer state.
        update_fn: a function that can perform the update.
    '''
    optax_opt = optax_optimizer(*optax_args, **optax_kwargs)
    # adamw = optax.adamw(learning_rate=1.0, b1=beta1, b2=beta2, weight_decay=wd, eps=epsilon)
    if per_variable_lr:
        init_ol_param = tree_map(
            lambda x: jnp.zeros(1),
            params
        )

    else:
        init_ol_param = jnp.zeros(1)
    state = {
        'optax_state': optax_opt.init(params),
        'prev_update': zeros_like(params),
        'prev_lr_residual': zeros_like(init_ol_param),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'prev_scaling': zeros_like(init_ol_param),
        'clip': clip,
        'clip_meta_grad': clip_meta_grad,
        'ol_state':  ol_init(init_ol_param, *ol_args, **ol_kwargs), #ol_state, #ol_init(jnp.zeros(1), *ol_args, **ol_kwargs),
        'steps_per_ol_update': steps_per_ol_update,
        'ol_grad_accumulator': zeros_like(init_ol_param),
        'true_params': params,
        'prev_true_params': params,
        'num_ol_steps': 0,
        'last_ol_logs': {},
        'loss_accumulator': 0,
    }
    return state, functools.partial(
        optax_learned_lr_update,
        multiply,
        additive_bounds,
        use_loss_diff,
        use_rand_scaling,
        per_variable_lr,
        optax_opt,
        ol_update_fn)

def optax_learned_lr_update(
    multiply,
    additive_bounds,
    use_loss_diff,
    use_rand_scaling,
    per_variable_lr,
    optax_opt,
    ol_update_fn,
    loss_and_grad_fn,
    rng,
    model_and_example_state,
    opt_state,
    lr=jnp.array(1.0),
    do_logging=False):
    '''
    update function for learning optax lrs

    arguments:
        multiply: if true, learn a multiplicative residual: lr = base_lr * exp(residual). If false, do an additive lr = base_lr + residual
        additive_bounds: if true, change the behavior of lower and upper bounds to be base_lr - lower_bound and base_lr + upper_bound.
                         This option is only supported in combination with multiply=False
        use_loss_diff: if true, use the loss difference instead of the gradient, update inner product.
        use_rand_scaling: if true, then we scale the residuals by a random uniform [0,1] value.
        per_variable_lr: if true, learn a separate lr scaling (or residual) for each model variable (not each coordinate)!
        optax_opt: optax optimizer object.
        ol_update_fn: update function for the online learner.
            *** these first four arguments will be set by optax_learned_lr_init via a partial because they cannot be jitted (two of them
                are not JAX types, and the other two are flags that are checked via python conditionals.)
        
        loss_and_grad_fn: a function that takes as input a pytree containing model state and minibatch example info and returns a loss, grad tuple.
            *** this one is probably not a JAX type, so to jit you need to make it a static argument or wrap in a partial!

        rng: JAX rng
        model_and_example_state: input pytree to loss_and_grad_fn.
            This is assumed to be a dictionary containing a key 'model_state'.
            model_and_example_state['model_state'] should be the state of a Flax model, and so should contain two
            keys: 'params' and 'constants'. model_and_example_state['model_state']['params'] are the parameters to be
            optimized. 'constants' are just constants (i.e. constant attention masks or other buffers).
        opt_state: state of the optimizer (pytree).
        lr: current learning rate.

        do_logging: a flag that tells whether to output some additional logging info in a dictionary.
            *** This is checked via python conditional, so it should probably be a "static" argument when jitting or wrapped in a partial!

    returns:
        rng: next value for JAX rng
        value: loss value from loss_and_grad_fn. Should have at least one key "loss" that holds the loss.
        grad: gradient value from loss_and_grad_fn
        model_state_next: next value for model_state (i.e this will be the next value for model_and_example_state['model_state'])
        opt_state_next: next value for optimizer state (i.e. this will be the next value for opt_state)
        log_dict: dictionary of logging info. Is set to None if do_logging=False.
    '''


    
    optax_state = opt_state['optax_state']
    prev_update = opt_state['prev_update']
    prev_lr_residual = opt_state['prev_lr_residual']
    ol_state = opt_state['ol_state']
    lower_bound = opt_state['lower_bound']
    upper_bound = opt_state['upper_bound']
    prev_scaling = opt_state['prev_scaling']
    clip = opt_state['clip']
    clip_meta_grad = opt_state['clip_meta_grad']
    ol_grad_accumulator = opt_state['ol_grad_accumulator']
    steps_per_ol_update = opt_state['steps_per_ol_update']
    num_ol_steps = opt_state['num_ol_steps']
    true_params = opt_state['true_params']
    prev_true_params = opt_state['prev_true_params']
    last_ol_logs = opt_state['last_ol_logs']
    loss_accumulator = opt_state['loss_accumulator']



    value, grad = loss_and_grad_fn(model_and_example_state)


    model_state = model_and_example_state['model_state']

    params = model_state['params']
    constants = model_state['constants']



    model_and_example_state['model_state']['params'] = prev_true_params

    value_prev, _ = loss_and_grad_fn(model_and_example_state)

    model_and_example_state['model_state']['params'] = true_params

    value_true, _ = loss_and_grad_fn(model_and_example_state)


    loss_diff = value_true['loss'] - value_prev['loss']

    loss_accumulator_next = loss_accumulator + loss_diff


    num_ol_steps_next = num_ol_steps + 1



    # compute gradient with respect to residual
    if per_variable_lr:
        res_grad_current = tree_map(
            lambda u, g: jnp.sum(u * g),
            prev_update,
            grad
        )
    else:
        res_grad_current = tree_dot(prev_update, grad)



    if use_loss_diff:
        if per_variable_lr:
            total_dot_product = tree_reduce(
                lambda ac, x: ac + x,
                res_grad_current)
            res_grad_scaling = (loss_diff/(1e-8 + total_dot_product * ol_state['prediction'])) 
            res_grad_current = tree_map(
                lambda x: x * res_grad_scaling,
                res_grad_current
            )
        else:
            res_grad_current = (loss_diff/(1e-8 + ol_state['prediction'])) 


    res_grad = tree_map(
        lambda a, r: a + r,
        ol_grad_accumulator,
        res_grad_current
    )


    if multiply:
        res_grad = tree_map(
            lambda x: x*jnp.exp(prev_scaling * prev_lr_residual),
            res_grad
        )


    # clip gradients
    grad = tree_map(
        lambda g: g * jnp.minimum(1.0, clip/(1e-8+ jnp.linalg.norm(g))),
        grad
    )

    # clip res_grad
    res_grad = tree_map(
        lambda x: x * jnp.minimum(1.0, clip_meta_grad/(1e-8 + jnp.linalg.norm(x))),
        res_grad
    )


    cur_rng, rng  = jax.random.split(rng)

    if use_rand_scaling:
        rand_scaling = jax.random.uniform(cur_rng)
    else:
        rand_scaling = 1.0


    # recover previous prediction. Note that this is the UNCONSTRAINED residual
    ol_prediction = ol_state['prediction']

    # apply constraint set reduction for 1-D intervals.
    # this modifies the residual gradient to correctly preserve regret bounds
    # over clipping (note that simply differentiating through the clipping operation
    # does NOT preserve regret as it would zero-out the gradient whenever clipping occurs).
    res_grad = tree_map(
        lambda ol_p, prev_res, r_g: jnp.where(
            -(ol_p - prev_res) * jnp.sign(r_g) > 1e-8, # should be == 0 when no clipping happens, but idk about floating point stuff.
            jnp.zeros_like(r_g),
            r_g),
        ol_prediction,
        prev_lr_residual,
        res_grad)


    # perform online learning update on residual. The residual will be the "prediction" of the online learner.
    ol_state_next, ol_logs = ol_update_fn(res_grad, ol_state) #jax.lax.cond(num_ol_steps_next >= steps_per_ol_update, lambda x: ol_update_fn(res_grad, ol_state), lambda x: ol_state, last_ol_logs)


    # clip residual to within bounds

    ol_prediction_next = jax.lax.cond(num_ol_steps_next >= steps_per_ol_update, lambda : ol_state_next['prediction'], lambda : ol_state['prediction'])
    ol_state_next = jax.lax.cond(num_ol_steps_next >= steps_per_ol_update, lambda : ol_state_next, lambda : ol_state)
    ol_grad_accumulator_next = tree_map(lambda x: (num_ol_steps_next < steps_per_ol_update) * x, res_grad)
    loss_accumulator_next = tree_map(lambda x: (num_ol_steps_next < steps_per_ol_update) * x, loss_accumulator_next)
    num_ol_steps_next = (num_ol_steps_next < steps_per_ol_update) * num_ol_steps_next

    if multiply:
        residual = tree_map(
            lambda o: jnp.clip(o, a_min=jnp.log(lower_bound), a_max=jnp.log(upper_bound)),
            ol_prediction_next)
    else:
        if additive_bounds:
            residual = tree_map(
                lambda o: jnp.clip(o, a_min=lower_bound, a_max=upper_bound),
                ol_prediction_next)
        else:
            residual = tree_map(
                lambda o: jnp.clip(o, a_min=lower_bound*lr - lr, a_max=upper_bound*lr - lr),
                ol_prediction_next)
    

    update, optax_state_next = optax_opt.update(grad,  optax_state, params)


    if multiply:
        learned_lr = tree_map(
            lambda r: lr * jnp.exp(rand_scaling * r),
            residual)

        true_params_next = tree_map(
            lambda p, u, r: p + lr * jnp.exp(r) * u,
            true_params,
            update,
            residual
        )

        update = tree_map(
            lambda u, r: lr * jnp.exp(rand_scaling * r) * u,
            update,
            residual)
        # multiply e^residual by lr
        param_next = tree_map(
            lambda p, u: p + u,
            params,
            update)


    else:
        true_params_next = tree_map(
            lambda p, u, r: p + (lr + r) * u,
            true_params,
            update,
            r
        )
        learned_lr = tree_map(
            lambda r: lr + rand_scaling * r,
            residual
        )
        # add mode: add residual with random scaling to the lr
        param_next = tree_map(
            lambda p, u: p + learned_lr * u,
            true_params,
            update)



    opt_state_next = {
        'optax_state': optax_state_next,
        'prev_update': update,
        'prev_lr_residual': residual,
        'prev_scaling': rand_scaling,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'ol_state': ol_state_next,
        'clip': clip,
        'clip_meta_grad': clip_meta_grad,
        'ol_grad_accumulator': ol_grad_accumulator_next,
        'num_ol_steps': num_ol_steps_next,
        'steps_per_ol_update': steps_per_ol_update,
        'last_ol_logs': ol_logs,
        'true_params': true_params_next,
        'prev_true_params': true_params,
        'loss_accumulator': loss_accumulator_next,
    }

    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {
            'learned_lr': tree_average(learned_lr),
            # 'expected_learned_lr': lr + 0.5*residual,
            'ol_prediction': tree_average(ol_state_next['prediction']),
            'residual': tree_average(residual),
            'grad_accumulator': tree_average(ol_grad_accumulator_next),
            'loss_accumulator': loss_accumulator_next,
        }
        log_dict.update(ol_logs)
    else:
        log_dict = None

    return rng, value, grad, model_state_next, opt_state_next, log_dict



def adamw_learned_lr_init(params, ol_init, beta1=0.9, beta2=0.99, wd=0.0, epsilon=1e-8, lower_bound=1e-8, upper_bound=10, *args, **kwargs):
    state = {
        'beta1': jnp.array(beta1),
        'beta2': jnp.array(beta2),
        'wd': jnp.array(wd),
        'epsilon': jnp.array(epsilon),
        'count': jnp.array(0),
        'm': zeros_like(params),
        'v': zeros_like(params),
        'prev_update': zeros_like(params),
        'prev_lr_residual': jnp.zeros(1),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'max_lr': jnp.array(0.0),
        'ol_state': ol_init(jnp.zeros(1), *args, **kwargs),
    }
    return state




def adamw_learned_lr_update(ol_update, loss_and_grad_fn, rng, model_and_example_state, opt_state, lr=jnp.array(1.0), do_logging=False):
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
            lr: scaling value (e.g. a learning rate).

            do_logging: flag for whether to output logging info
        
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
    beta1 = opt_state['beta1']
    beta2 = opt_state['beta2']
    wd = opt_state['wd']
    epsilon = opt_state['epsilon']
    count = opt_state['count']
    prev_update = opt_state['prev_update']
    prev_lr_residual = opt_state['prev_lr_residual']
    ol_state = opt_state['ol_state']
    lower_bound = opt_state['lower_bound']
    upper_bound = opt_state['upper_bound']
    max_lr = opt_state['max_lr']


    ol_prediction = ol_state['prediction']


    # compute gradient with respect to residual
    res_grad = tree_dot(prev_update, grad)


    cur_rng, rng  = jax.random.split(rng)

    rand_scaling = jax.random.uniform(cur_rng)


    # apply constraint set reduction for 1-D intervals.
    # this modifies the residual gradient to correctly preserve regret bounds
    # over clipping (note that simply differentiating through the clipping operation
    # does NOT preserve regret as it would zero-out the gradient whenever clipping occurs).
    res_grad = jnp.where(
        -(ol_prediction - prev_lr_residual) * jnp.sign(res_grad) > 1e-8, # should be == 0 when no clipping happens, but idk about floating point stuff.
        jnp.zeros_like(res_grad),
        res_grad)


    # perform online learning update on residual. The residual will be the "prediction" of the online learner.
    ol_state, ol_logs = ol_update(res_grad, ol_state)


    # clip residual to within bounds
    residual = jnp.clip(ol_state['prediction'], a_min=lower_bound*lr - lr, a_max=upper_bound*lr - lr)
    

    # compute adamw update
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


    max_lr_next = jnp.maximum(max_lr, lr + rand_scaling * residual)

    # add residual with random scaling to the lr
    param_next = tree_map(
        lambda p, u: p + (lr + rand_scaling * residual) * u,
        params,
        update)



    opt_state_next = {
        'beta1': beta1,
        'beta2': beta2,
        'wd': wd,
        'epsilon': epsilon,
        'count': count_next,
        'm': m_next,
        'v': v_next,
        'prev_update': update,
        'prev_lr_residual': residual,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'ol_state': ol_state,
        'max_lr': max_lr_next,
    }

    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {
            'learned_lr': lr + rand_scaling*residual,
            'expected_learned_lr': lr + 0.5*residual,
            'max_learned_lr': max_lr_next,
            'residual': residual,
            'unconstrained_residual': jnp.average(ol_state['prediction'])
        }
        log_dict.update(ol_logs)
    else:
        log_dict = None

    return rng, value, grad, model_state_next, opt_state_next, log_dict



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


def adagrad_init(params, lr=1.0, eps=1e-8, decay=1.0):
    state = {
        'grad_squared_sum': zeros_like(params),
        'prediction': zeros_like(params),
        'decay': decay,
        'eps': eps,
        'lr': lr,
    }

    return state

def adagrad_reset(old_state):
    state = {
        'grad_squared_sum': zeros_like(old_state['grad_squared_sum']),
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'eps': old_state['eps'],
        'lr': old_state['lr'],
    }

    return state    

def adagrad_update(grad, opt_state, do_logging=False):

    grad_squared_sum = opt_state['grad_squared_sum']
    prediction = opt_state['prediction']
    decay = opt_state['decay']
    eps = opt_state['eps']
    lr = opt_state['lr']

    grad_squared_sum_next = tree_map(
        lambda s, g: s * decay + g**2,
        grad_squared_sum,
        grad
    )

    prediction_next = tree_map(
        lambda p, g, s: p - lr * g / (eps + jnp.sqrt(s)),
        prediction,
        grad,
        grad_squared_sum
    )

    opt_state_next = {
        'grad_squared_sum': grad_squared_sum_next,
        'prediction': prediction_next,
        'decay': decay,
        'eps': eps,
        'lr': lr,
    }

    return opt_state_next, {}



def cb_init(params, eps=1.0, eta=2.0/(2-np.log(3)), decay=1.0, **ignored_kwargs):
    state = {}
    state['wealth'] = zeros_like(params)
    state['bet_fractions'] = zeros_like(params)
    state['bet_grad_squared_sum'] = zeros_like(params)
    state['max_grads'] = zeros_like(params)
    state['prediction'] = zeros_like(params)
    state['grad_sum'] = zeros_like(params)
    state['eta'] = eta
    state['eps'] = eps
    state['decay'] = decay

    return state


def cb_reset(old_state, count):
    state = {
        'wealth': zeros_like(old_state['wealth']),
        'bet_fractions': zeros_like(old_state['bet_fractions']),
        'bet_grad_squared_sum': zeros_like(old_state['bet_grad_squared_sum']),
        'max_grads': old_state['max_grads'],
        'prediction': zeros_like(old_state['prediction']),
        'grad_sum': zeros_like(old_state['grad_sum']),
        'eta': old_state['eta'],
        'eps': old_state['eps']/(count + 1),
        'decay': old_state['decay'],
    }
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
    wealth_next = tree_map(lambda r, g, b, m: r * decay - b * g * (r+ m*eps), wealth, bet_fractions, grad, max_grads)
    
    
    bet_fractions_next = tree_map(
        lambda b, z, s, m: jnp.clip(
            b - z * eta/(1e-8 + s),
            a_min=-0.5/(1e-8 + m), a_max=0.5/(1e-8 + m)),
        bet_fractions,
        bet_grad,
        bet_grad_squared_sum_next,
        max_grads_next)

    param_next = tree_map(lambda r, b, m: (eps * m + r) * b, wealth_next, bet_fractions_next, max_grads_next)

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



# TODO: make this less repetitive....

def cb_stable_init(params, eps=1.0, eta=1.0, decay=1.0, stability=0.0, grad_stab=0.0):
    state = {
        'wealth': tree_map(lambda x: jnp.full_like(x, fill_value=eps), params), #zeros_like(params),
        'bet_fractions': zeros_like(params),
        'subepoch_grad': zeros_like(params),
        'max_grads': zeros_like(params),
        'subepoch_grad_abs_sum': zeros_like(params),
        'subepoch_grad_sum': zeros_like(params),
        'eps': eps,
        'decay': decay,
        'stability': stability,
        'grad_stab': grad_stab,
        'eta': eta,
        'prediction': zeros_like(params),
        'subepoch_count': zeros_like(params),
    }
    # state['wealth'] = zeros_like(params)
    # state['bet_fractions'] = zeros_like(params)
    # state['grad_squared_sum'] = zeros_like(params)
    # state['max_grads'] = zeros_like(params)
    # state['prediction'] = zeros_like(params)
    # state['grad_sum'] = zeros_like(params)
    # state['eta'] = eta
    # state['eps'] = eps
    # state['decay'] = decay

    return state


def cb_stable_reset(old_state, count):
    state = {
        'wealth': tree_map( lambda x: jnp.full_like(x, fill_value=old_state['eps']/(count+1)), old_state['wealth']),
        'bet_fractions': zeros_like(old_state['bet_fractions']),
        'subepoch_grad': zeros_like(old_state['subepoch_grad']),
        'max_grads': old_state['max_grads'],
        'subepoch_grad_abs_sum': zeros_like(old_state['subepoch_grad_abs_sum']),
        'subepoch_grad_sum': zeros_like(old_state['subepoch_grad_sum']),
        'eta': old_state['eta'],
        'eps': old_state['eps']/(count + 1),
        'decay': old_state['decay'],
        'stability': old_state['stability'],
        'grad_stab': old_state['grad_stab'],
        'prediction': zeros_like(old_state['prediction']),
        'subepoch_count': zeros_like(old_state['subepoch_count']),
    }
    return state

def cb_stable_update(grad, opt_state, do_logging=False):
    
    wealth = opt_state['wealth']
    bet_fractions = opt_state['bet_fractions']
    subepoch_grad = opt_state['subepoch_grad']
    max_grads = opt_state['max_grads']
    eps = opt_state['eps']
    decay = opt_state['decay']
    stability = opt_state['stability']
    grad_stab = opt_state['grad_stab']
    subepoch_grad_abs_sum = opt_state['subepoch_grad_abs_sum']
    subepoch_grad_sum = opt_state['subepoch_grad_sum']
    eta = opt_state['eta']
    subepoch_count = opt_state['subepoch_count']



    current_stab = tree_map(lambda m: stability + grad_stab * m, max_grads)

    max_grads_next = tree_map(lambda m, g: jnp.maximum(m * decay, jnp.abs(g)), max_grads, grad)

    grad = tree_map(lambda g, m: jnp.clip(g, a_min=-m, a_max=m), grad, max_grads)

    subepoch_grad_next = tree_map(lambda s, g: s + g, subepoch_grad, grad)

    end_subepoch_mask = tree_map(lambda s, t: jnp.abs(s) > t, subepoch_grad_next, current_stab)

    subepoch_count_next = tree_map(lambda c, m: c + m, subepoch_count, end_subepoch_mask)

    subepoch_grad_abs_sum_next = tree_map(
        lambda s, m, g: s * decay + m * jnp.abs(g),
        subepoch_grad_abs_sum,
        end_subepoch_mask,
        subepoch_grad_next)

    subepoch_grad_sum_next = tree_map(
        lambda s, m, g: s*decay + m * g,
        subepoch_grad_sum,
        end_subepoch_mask,
        subepoch_grad_next)

    bet_fractions_next = tree_map(
        lambda s, d, t, m, b, mask, : jnp.where(mask, -eta * s/(1e-8 + 2 * (m+t) * (t+m + d)), b),
        subepoch_grad_sum_next,
        subepoch_grad_abs_sum_next,
        current_stab,
        max_grads_next,
        bet_fractions,
        end_subepoch_mask,
    )

    wealth_next = tree_map(
        lambda w, b, g, m, bn, c: jnp.where(
            m,
            w * ( (decay - b*(g+ jnp.sign(g) * c))/(1.0 - c * jnp.sign(g) * bn)),
            w),
        wealth,
        bet_fractions,
        subepoch_grad_next,
        end_subepoch_mask,
        bet_fractions_next,
        current_stab,
    )

    prediction = tree_map(
        lambda w, b: w*b,
        wealth_next,
        bet_fractions_next,
    )



    subepoch_grad_next = tree_map(
        lambda s, m: s * jnp.logical_not(m),
        subepoch_grad_next,
        end_subepoch_mask
    )

    state = {
        'wealth': wealth_next,
        'bet_fractions': bet_fractions_next,
        'subepoch_grad': subepoch_grad_next,
        'subepoch_grad_sum': subepoch_grad_sum_next,
        'max_grads': max_grads_next,
        'subepoch_grad_abs_sum': subepoch_grad_abs_sum_next,
        'eps': eps,
        'decay': decay,
        'stability': stability,
        'grad_stab': grad_stab,
        'eta': eta,
        'prediction': prediction,
        'subepoch_count': subepoch_count_next,
    }

    return state, {
        'wealth': tree_map(lambda x: jnp.average(x), wealth_next),
        'bet_fractions': tree_map(lambda x: jnp.average(x), bet_fractions_next),
        'max_grad': tree_map(lambda x: jnp.average(x), max_grads_next),
        'subepoch_grad_sum': tree_map(lambda x: jnp.average(x), subepoch_grad_sum_next),
        'subepoch_grad': tree_map(lambda x: jnp.average(x), subepoch_grad_next),
        'subepoch_count': tree_map(lambda x: jnp.average(x), subepoch_count_next),
        }


    

def OL_momentum_init(params, ol_init, ol_args, ol_kwargs, ol_update_fn, ol_reset_fn, reset_threshold=100.0,):
    state = {
        'ol_state': ol_init(params, *ol_args, **ol_kwargs),
        'true_params': params,
        'last_offset': zeros_like(params),
        'total_reward': 0.0,
        'epoch_reward': 0.0,
        'epoch_count': 0,
        'iteration_count':  0,
        'reset_threshold': reset_threshold,
    }

    return state, functools.partial(OL_momentum_update, ol_update_fn, ol_reset_fn)

def OL_momentum_update(ol_update, ol_reset, loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):

    value, grad = loss_and_grad_fn(model_and_example_state)
    model_state = model_and_example_state['model_state']

    cur_rng, rng = jax.random.split(rng)

    rand_scaling = jax.random.uniform(cur_rng)
 
    constants = model_state['constants']

    ol_state = opt_state['ol_state']
    true_params = opt_state['true_params']
    total_reward = opt_state['total_reward']
    epoch_reward = opt_state['epoch_reward']
    last_offset = opt_state['last_offset']
    epoch_count = opt_state['epoch_count']
    iteration_count = opt_state['iteration_count']
    reset_threshold = opt_state['reset_threshold']

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

    ol_state, epoch_reward_next, epoch_count_next = jax.lax.cond(
        jnp.all(epoch_reward_next > reset_threshold),
        lambda s, c: (ol_reset(s, c), jnp.zeros(1), c + 1),
        lambda s, c: (s, jnp.zeros(1), c),
        ol_state, epoch_count)

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
        'epoch_reward': epoch_reward_next,
        'iteration_count': iteration_count_next,
        'epoch_count': epoch_count_next,
        'reset_threshold': reset_threshold,
        'last_offset': offset,
    }


    if do_logging:
        log_dict = {
            'total_reward': total_reward_next,
            'epoch_reward': epoch_reward_next,
            'epoch_count': epoch_count_next
        }
        # for key, value in ol_logs.items():
        #     log_dict['ol_'+key] = value
    else:
        log_dict = None


    return rng, value, grad, model_state_next, opt_state_next, log_dict

    
   





# def OL_momentum_ogd_init(params, ol_lr):
#     state = {
#         'ogd_state': ogd_init(params, lr=ol_lr),
#         'true_params': params
#     }
#     return state

# def OL_momentum_ogd_update(loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):

#     value, grad = loss_and_grad_fn(model_and_example_state)
#     model_state = model_and_example_state['model_state']

#     cur_rng, rng = jax.random.split(rng)

#     scaling = jax.random.uniform(cur_rng)

#     params = model_state['params']
#     constants = model_state['constants']

#     ogd_state = opt_state['ogd_state']
#     true_params = opt_state['true_params']

#     offset, ogd_state_next = ogd_update(grad, ogd_state, min_bound=-scale, max_bound=scale)

#     params_next = tree_map(lambda p, o: p+ scaling * o, true_params, offset)


#     model_state_next = {
#         'constants': constants,
#         'params': params_next
#     }

#     opt_state_next = {
#         'ogd_state': ogd_state_next,
#         'true_params': tree_map(lambda p, o: p+ o, true_params, offset)
#     }


#     if do_logging:
#         log_dict = {}
#     else:
#         log_dict = None


#     return rng, value, grad, model_state_next, opt_state_next, log_dict

    


# def OL_momentum_expmd_init(params, grid_size=10, min_eta=1e-8, max_eta=1.0):
#     ol_state = exp_md_pm_init(params, grid_size, min_eta, max_eta)
#     return ol_state

# def OL_momentum_expmd_update(loss_and_grad_fn, rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False):

#     value, grad = loss_and_grad_fn(model_and_example_state)
#     model_state = model_and_example_state['model_state']

#     params = model_state['params']
#     constants = model_state['constants']

#     offset, opt_state_next = exp_md_pm_update(grad, opt_state, min_bound=-scale, max_bound=scale)

#     param_next = tree_map(lambda p, o: p+o, params, offset)


#     model_state_next = {
#         'constants': constants,
#         'params': param_next
#     }


#     if do_logging:
#         log_dict = {}
#     else:
#         log_dict = None


#     return rng, value, grad, model_state_next, opt_state_next, log_dict

    









def exp_md_pos_init(params, grid_size=10, min_eta=1e-8, max_eta=1.0):
    thetas = [zeros_like(params) for _ in range(grid_size)]
    iterates = [zeros_like(params) for _ in range(grid_size)]

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




# rng = jax.random.PRNGKey(0)
# loss_fn = lambda x:  jnp.sum(jnp.zeros(2) * x['r'])

# val_and_grad = jax.value_and_grad(loss_fn)

# def loss_and_grad_fn(model_and_example_state):
#     params = model_and_example_state['model_state']['params']
#     return val_and_grad(params)

# OL_momentum_expmd_init

# model_state = {
#     'constants': {'c': jnp.ones(2)},
#     'params': {'r': jnp.array([2.0,3.0])}
# }

# model_and_example_state = {
#     'model_state': model_state
# }



# opt_state = OL_momentum_init(model_state['params'], ol_init=cb_init, decay=0.99)

# OL_momentum_update_jit = jax.jit(functools.partial(OL_momentum_update, cb_update, loss_and_grad_fn), static_argnames='do_logging')

# # OL_momentum_update_jit =  functools.partial(OL_momentum_update, cb_update, loss_and_grad_fn)


# for _ in range(1):
#     rng, value, grad, model_state, opt_state, log_dict = OL_momentum_update_jit(rng, model_and_example_state, opt_state, scale=jnp.array(1.0), do_logging=False)
#     model_and_example_state = {
#         'model_state': model_state
#     }


# print(model_state)
# print(opt_state)


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
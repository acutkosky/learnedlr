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

def tree_norm(tree):
    return jnp.sqrt(tree_reduce(
        lambda a, x: a + x,
        tree_map(lambda z: jnp.sum(z**2), tree)
    ))

def clip_by_global_norm(tree, clip):
    norm = tree_norm(tree)
    return tree_map(
        lambda x: x/(1e-8 + norm),
        tree
    )

def clip_by_variable_norm(tree, clip):
    return tree_map(
        lambda x: x/(1e-8 + jnp.linalg.norm(x)),
        tree
    )
    
def tree_dot(a, b):
    return tree_reduce(
        lambda s, x: s + x,
        tree_map(
            lambda x, y: jnp.sum(x * y),
            a,
            b
        )
    )







#### ONLINE LEARNERS ####
#### Each has three functions:
#### _init takes the parameters and some hyperparams and returns the online learner state
#### _reset resets the state (basically like calling init but without the parameters).
#### _update takes a gradient and the state and computes the new state.
####     state['prediction'] contains the current parameters predicted by the online learner.
#### CB_STABLE IS THE MOST ADVANCED ONE

def ogd_init(params, lr=1.0, eps=1e-8, decay=1.0):
    state = {
        'prediction': zeros_like(params),
        'decay': decay,
        'eps': eps,
        'lr': lr,
    }

    return state

def ogd_reset(old_state, epoch_count):
    state = {
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'lr': old_state['lr']/(epoch_count + 1.0),
    }

    return state    

def ogd_update(grad, opt_state, do_logging=False):

    prediction = opt_state['prediction']
    decay = opt_state['decay']
    lr = opt_state['lr']

    prediction_next = tree_map(
        lambda p, g: p * decay - lr * g,
        prediction,
        grad,
    )

    opt_state_next = {
        'prediction': prediction_next,
        'decay': decay,
        'lr': lr,
    }

    return opt_state_next, {}




def adagrad_init(params, lr=1.0, eps=1e-8, decay=1.0):
    state = {
        'grad_squared_sum': zeros_like(params),
        'prediction': zeros_like(params),
        'decay': decay,
        'eps': eps,
        'lr': lr,
    }

    return state

def adagrad_reset(old_state, epoch_count):
    state = {
        'grad_squared_sum': zeros_like(old_state['grad_squared_sum']),
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'eps': old_state['eps'],
        'lr': old_state['lr']/(epoch_count + 1.0),
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
        lambda p, g, s: p * decay - lr * g / (eps + jnp.sqrt(s)),
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




def cb_stable_init(params, eps=1e-2, eta=1.0, decay=0.999, stability=0.0, grad_stab=2.0):
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
        lambda s, mk, g: s * decay + mk * jnp.abs(g),
        subepoch_grad_abs_sum,
        end_subepoch_mask,
        subepoch_grad_next)

    subepoch_grad_sum_next = tree_map(
        lambda s, mk, g: s*decay + mk * g,
        subepoch_grad_sum,
        end_subepoch_mask,
        subepoch_grad_next)

    bet_fractions_next = tree_map(
        lambda s, gs, t, mx, : -eta * s/(1e-8 + 2 * (mx+t) * (t+mx + gs)),
        subepoch_grad_sum_next,
        subepoch_grad_abs_sum_next,
        current_stab,
        max_grads_next,
    )

    wealth_next = tree_map(
        lambda w, b, g, mk, bn, s: w * ( (decay - b*(g * mk+ jnp.sign(g * mk) * s))/(1.0 - s * jnp.sign(g * mk) * bn)),
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




########## END ONLINE LEARNERS ###############







########## Optax optimizer learning rate learners
### these have two functions:
### _init takes params and hyperparameters and returns a state and an update function.
###    the update function has a long signature. Take a look at
### _update to see what it is. _init will take care of automatically supplying some of the
### arguments to the update function via a partial. **you probably never want to directly call
###  _update - instead use the function returned by _init.

### there are two optimizers, first simply scales the lr of an optax optimizer by a random quantity.
### second (optax_ol_scaled) uses an online learner to choose the lr.


def optax_rand_scaled_init(
    params,
    optax_optimizer,
    optax_args,
    optax_kwargs,
    clip=1.0,
    rand_scaling_type='uniform',
    use_loss_diff=False,
    **_kwargs):
    '''
    scales an optax optimizer by a random amount.

    arguments:
        params: model parameters (pytree)

        optax_optimizer: optimizer class from optax to use as base update generator.
        optax_args: argument list for optax optimizer
        optax_kwargs: keyword argument dict for optax optimizer

        clip: clip each variable of the gradient to have this norm
        
        rand_scaling_type: string, specifies which type of random scaling.
            can be 'none' (just regular adamw), 'uniform' (uniform in [0,1]),
            or 'exponential' (from an exponential distribution with mean 1),
            or 'half': scale by 0.5

        use_loss_diff: if true, we will rescale the gradient so that
            <gradient , update> = loss(param + update) - loss(param)


    returns:
        state: pytree optimizer state.
        update_fn: a function that can perform the update.
    '''
    optax_opt = optax_optimizer(*optax_args, **optax_kwargs)
    state = {
        'optax_state': optax_opt.init(params),
        'clip': clip,
        'prev_update': zeros_like(params),
        'true_params': params,
        'prev_true_params': params,
        'prev_rand_update_scaling': jnp.ones(1),
        'prev_true_update_scaling': jnp.ones(1)
    }
    return state, functools.partial(
        optax_rand_scaled_update,
        rand_scaling_type,
        use_loss_diff,
        optax_opt)

def optax_rand_scaled_update(
    rand_scaling_type,
    use_loss_diff,
    optax_opt,
    loss_and_grad_fn,
    rng,
    model_and_example_state,
    opt_state,
    lr=jnp.array(1.0),
    do_logging=False):
    '''
    update function for random scaling of optax lrs

    arguments:
        rand_scaling_type: string, specifies which type of random scaling.
            can be 'none' (just regular adamw), 'uniform' (uniform in [0,1]),
            or 'exponential' (from an exponential distribution with mean 1),
            or 'half': scale by 0.5
        optax_opt: optax optimizer object.
        use_loss_diff: if true, we will rescale the gradient so that
            <gradient , update> = loss(param + update) - loss(param)
            *** these first arguments will be set by optax_rand_scaled_init via a partial because they cannot be jitted.
            *** (e.g. some are options checked by python conditionals)
        
        loss_and_grad_fn: a function that takes as input a pytree containing model state and minibatch example info and returns a 
            tuple (value, grad). "value" should be a dict that has at least one key 'loss'
            containing the loss value (this key is only used if use_loss_diff is True).
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
    clip = opt_state['clip']
    true_params = opt_state['true_params']
    prev_true_params = opt_state['prev_true_params']
    prev_update = opt_state['prev_update']
    prev_rand_update_scaling = opt_state['prev_rand_update_scaling']
    prev_true_update_scaling = opt_state['prev_true_update_scaling']


    model_state = model_and_example_state['model_state']

    params = model_state['params']
    constants = model_state['constants']


    value, grad = loss_and_grad_fn(model_and_example_state)




    if use_loss_diff:
        model_and_example_state['model_state']['params'] = true_params

        true_loss, _ = loss_and_grad_fn(model_and_example_state)


        model_and_example_state['model_state']['params'] = prev_true_params


        prev_true_loss, _ = loss_and_grad_fn(model_and_example_state)

        loss_diff = true_loss['loss'] - prev_true_loss['loss']

        grad_update_product = tree_dot(grad, prev_update)
        grad = tree_map(
            lambda g: g * (1e-5 + jnp.abs(loss_diff)) / (1e-8 + jnp.abs(grad_update_product)),
            grad)

    

    # clip gradients
    grad = clip_by_variable_norm(grad, clip)


    update, optax_state_next = optax_opt.update(grad,  optax_state, params)


    if rand_scaling_type == 'uniform':
        cur_rng, rng  = jax.random.split(rng)
        rand_scaling = jax.random.uniform(cur_rng)
        rand_scaled_lr = rand_scaling * lr
        true_scaled_lr = lr
    elif rand_scaling_type == 'exponential':
        cur_rng, rng  = jax.random.split(rng)
        rand_scaling = -jnp.log(jax.random.uniform(cur_rng))
        rand_scaled_lr = rand_scaling * lr
        true_scaled_lr = rand_scaling * lr # the point of exponential is that these are the same!
    elif rand_scaling_type == 'half':
        rand_scaled_lr = 0.5 * lr
        true_scaled_lr = lr
    elif rand_scaling_type == 'none':        
        rand_scaled_lr = lr
        true_scaled_lr = lr
    else:
        raise(NotImplementedError, "unknown rand scaling!")


    param_next = tree_map(
        lambda p, u: p + rand_scaled_lr * u,
        true_params,
        update)

    true_params_next = tree_map(
        lambda p, u: p + true_scaled_lr * u,
        true_params,
        update)



    opt_state_next = {
        'optax_state': optax_state_next,
        'prev_update': update,
        'prev_rand_update_scaling': rand_scaled_lr,
        'prev_true_update_scaling': true_scaled_lr,
        'true_params': true_params_next,
        'prev_true_params': true_params,
        'clip': clip,
    }

    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {
            'rand_scaled_lr': rand_scaled_lr,
        }

        if use_loss_diff:
            log_dict['loss_diff'] = loss_diff
            log_dict['grad_update_product'] = grad_update_product
    else:
        log_dict = None

    return rng, value, grad, model_state_next, opt_state_next, log_dict






def optax_ol_scaled_init(
    params,
    optax_optimizer,
    optax_args,
    optax_kwargs,
    ol_update_fn=cb_stable_update,
    ol_init=cb_stable_init,
    ol_args=[],
    ol_kwargs={},
    clip=0.1,
    clip_ol=200.0,
    lower_bound=1e-8,
    upper_bound=1e-2,
    rand_scaling_type='uniform',
    use_loss_diff=False,
    **_kwargs):
    '''
    scales an optax optimizer by a random amount.

    arguments:
        params: model parameters (pytree)

        optax_optimizer: optimizer class from optax to use as base update generator.
        optax_args: argument list for optax optimizer
        optax_kwargs: keyword argument dict for optax optimizer

        ol_init: function that returns the initial state for an online learner.
        ol_args: argument list for ol_init
        ol_kwargs: keyword argument dict for ol_init.

        ol_update_fn: update function for the online learner.

        clip: clip each variable of the gradient to have this norm
        clip_ol: clip the meta-online learning gradient to have this norm

        lower_bound: lower bound for learned lr
        upper_bound: upper bound for learned lr
        
        rand_scaling_type: string, specifies which type of random scaling.
            can be 'none' (just regular adamw), 'uniform' (uniform in [0,1]),
            or 'exponential' (from an exponential distribution with mean 1),
            or 'half': scale by 0.5

        use_loss_diff: if true, we will rescale the gradient so that
            <gradient , update> = loss(param + update) - loss(param)


    returns:
        state: pytree optimizer state.
        update_fn: a function that can perform the update.
    '''
    optax_opt = optax_optimizer(*optax_args, **optax_kwargs)
    state = {
        'optax_state': optax_opt.init(params),
        'ol_state': ol_init(jnp.zeros(1), *ol_args, **ol_kwargs),
        'clip': clip,
        'clip_ol': clip_ol,
        'prev_update': zeros_like(params),
        'true_params': params,
        'prev_true_params': params,
        'prev_rand_update_scaling': jnp.ones(1),
        'prev_true_update_scaling': jnp.ones(1),
        'constraint_violation': jnp.zeros(1),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'prev_lr': jnp.zeros(1),
    }
    return state, functools.partial(
        optax_ol_scaled_update,
        rand_scaling_type,
        use_loss_diff,
        ol_update_fn,
        optax_opt)

def optax_ol_scaled_update(
    rand_scaling_type,
    use_loss_diff,
    ol_update,
    optax_opt,
    loss_and_grad_fn,
    rng,
    model_and_example_state,
    opt_state,
    lr=jnp.array(1.0),
    do_logging=False):
    '''
    update function for random scaling of optax lrs

    arguments:
        rand_scaling_type: string, specifies which type of random scaling.
            can be 'none' (just regular adamw), 'uniform' (uniform in [0,1]),
            or 'exponential' (from an exponential distribution with mean 1),
            or 'half': scale by 0.5
        use_loss_diff: if true, we will rescale the gradient so that
            <gradient , update> = loss(param + update) - loss(param)
        ol_update: update function for online learner
        optax_opt: optax optimizer object.
            *** these first arguments will be set by optax_rand_scaled_init via a partial because they cannot be jitted.
            *** (e.g. some are options checked by python conditionals)
        
        loss_and_grad_fn: a function that takes as input a pytree containing model state and minibatch example info and returns a 
            tuple (value, grad). "value" should be a dict that has at least one key 'loss'
            containing the loss value (this key is only used if use_loss_diff is True).
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
    clip = opt_state['clip']
    clip_ol = opt_state['clip_ol']
    true_params = opt_state['true_params']
    prev_true_params = opt_state['prev_true_params']
    prev_update = opt_state['prev_update']
    prev_rand_update_scaling = opt_state['prev_rand_update_scaling']
    prev_true_update_scaling = opt_state['prev_true_update_scaling']
    ol_state = opt_state['ol_state']
    constraint_violation = opt_state['constraint_violation']
    lower_bound = opt_state['lower_bound']
    upper_bound = opt_state['upper_bound']
    prev_lr = opt_state['prev_lr']


    model_state = model_and_example_state['model_state']

    params = model_state['params']
    constants = model_state['constants']


    value, grad = loss_and_grad_fn(model_and_example_state)




    if use_loss_diff:
        model_and_example_state['model_state']['params'] = true_params


        true_loss, _ = loss_and_grad_fn(model_and_example_state)


        model_and_example_state['model_state']['params'] = prev_true_params

        prev_true_loss, _ = loss_and_grad_fn(model_and_example_state)

        loss_diff = true_loss['loss'] - prev_true_loss['loss']

        ol_grad = loss_diff/(1e-8 + prev_lr)
    else:
        ol_grad = tree_dot(grad, prev_update)

    ol_grad = jnp.where(
            -constraint_violation * jnp.sign(ol_grad) > 1e-8, # should be == 0 when no clipping happens, but idk about floating point stuff.
            jnp.zeros_like(ol_grad),
            ol_grad)
    # clip ol_grad
    ol_grad = tree_map(
        lambda x: x * jnp.minimum(1.0, clip_ol/(1e-8 + jnp.linalg.norm(x))),
        ol_grad
    )
    
    ol_state_next, ol_logs = ol_update(ol_grad, ol_state, do_logging)

    unclipped_final_lr = lr + ol_state_next['prediction']

    final_lr = jnp.clip(unclipped_final_lr, a_min=lower_bound, a_max=upper_bound)

    constraint_violation_next = unclipped_final_lr - final_lr



    if rand_scaling_type == 'uniform':
        cur_rng, rng  = jax.random.split(rng)
        rand_scaling = jax.random.uniform(cur_rng)
        rand_scaled_lr = rand_scaling * final_lr
        true_scaled_lr = final_lr
    elif rand_scaling_type == 'exponential':
        cur_rng, rng  = jax.random.split(rng)
        rand_scaling = -jnp.log(jax.random.uniform(cur_rng))
        rand_scaled_lr = rand_scaling * final_lr
        true_scaled_lr = rand_scaling * final_lr # the point of exponential is that these are the same!
    elif rand_scaling_type == 'half':
        rand_scaled_lr = 0.5 * final_lr
        true_scaled_lr = final_lr
    elif rand_scaling_type == 'none':        
        rand_scaled_lr = final_lr
        true_scaled_lr = final_lr
    else:
        raise(NotImplementedError, "unknown rand scaling!")


    # clip gradients
    grad = clip_by_variable_norm(grad, clip)
    update, optax_state_next = optax_opt.update(grad,  optax_state, params)


    param_next = tree_map(
        lambda p, u: p + rand_scaled_lr * u,
        true_params,
        update)

    true_params_next = tree_map(
        lambda p, u: p + true_scaled_lr * u,
        true_params,
        update)



    opt_state_next = {
        'optax_state': optax_state_next,
        'ol_state': ol_state_next,
        'prev_update': update,
        'prev_rand_update_scaling': rand_scaled_lr,
        'prev_true_update_scaling': true_scaled_lr,
        'true_params': true_params_next,
        'prev_true_params': true_params,
        'clip': clip,
        'clip_ol': clip_ol,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'constraint_violation': constraint_violation_next,
        'prev_lr': final_lr
    }

    model_state_next = {
        'constants': constants,
        'params': param_next
    }


    if do_logging:
        log_dict = {
            'rand_scaled_lr': rand_scaled_lr,
            'learned_lr': final_lr,
            'unconstrained_residual': ol_state_next['prediction'],
        }

        log_dict.update(ol_logs)

        if use_loss_diff:
            log_dict['loss_diff'] = loss_diff
        
    else:
        log_dict = None

    return rng, value, grad, model_state_next, opt_state_next, log_dict




########## OL momentum based optimizers
#### these implement more precisely the theory-oriented algorithm
#### currently using with cb_stable uses a ton of slots.
#### I strongly suspect there is a more efficient implementation of cb_stable
#### that doesn't use as many slots.



def OL_momentum_init(params, ol_init, ol_args, ol_kwargs, ol_update_fn, ol_reset_fn, reset_threshold=100.0):
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

    
   




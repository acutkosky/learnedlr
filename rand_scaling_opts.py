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

def ones_like(tree):
    return tree_map(lambda x: jnp.ones_like(x), tree)

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
    scaling = jnp.minimum(1.0, clip/(1e-8 + norm))
    return tree_map(
        lambda x: scaling * x,
        tree
    )

def clip_by_variable_norm(tree, clip):
    return tree_map(
        lambda x: x * jnp.minimum(1.0, clip/(1e-8 + jnp.linalg.norm(x))),
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

def ogd_reset(old_state, epoch_count, do_decrease=True, reset_scaling=1.0):
    state = {
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'lr': reset_scaling * old_state['lr']*(epoch_count + 1)/(epoch_count + 2) if do_decrease else reset_scaling * old_state['lr'],
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

def adagrad_reset(old_state, epoch_count, do_decrease=True, reset_scaling=1.0):
    state = {
        'grad_squared_sum': zeros_like(old_state['grad_squared_sum']),
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'eps': old_state['eps'],
        'lr': reset_scaling * old_state['lr']*(epoch_count + 1)/(epoch_count + 2) if do_decrease else reset_scaling * old_state['lr'],
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
        grad_squared_sum_next
    )

    opt_state_next = {
        'grad_squared_sum': grad_squared_sum_next,
        'prediction': prediction_next,
        'decay': decay,
        'eps': eps,
        'lr': lr,
    }

    return opt_state_next, {'ol_lr': lr}


def simple_fr_init(params, base_lr=1.0, eta=1e-6, decay=1.0):
    state = {
        'grad_squared_sum': zeros_like(params),
        'prediction': zeros_like(params),
        'decay': decay,
        'eta': eta,
        'base_lr': base_lr,
        'grad_sum': zeros_like(params),
        'max_grad': zeros_like(params),
    }

    return state

def simple_fr_reset(old_state, epoch_count, do_decrease=True, reset_scaling=1.0):
    state = {
        'grad_squared_sum': zeros_like(old_state['grad_squared_sum']),
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'eta': old_state['eta'],
        'base_lr':  reset_scaling * old_state['base_lr']*(epoch_count + 1)/(epoch_count + 2) if do_decrease else reset_scaling * old_state['base_lr'],
        'grad_sum': zeros_like(old_state['grad_sum']),
        'max_grad': zeros_like(old_state['max_grad'])
    }

    return state   

def simple_fr_update(grad, opt_state, do_logging=False):

    grad_squared_sum = opt_state['grad_squared_sum']
    prediction = opt_state['prediction']
    decay = opt_state['decay']
    eta = opt_state['eta']
    base_lr = opt_state['base_lr']
    grad_sum = opt_state['grad_sum']
    max_grad = opt_state['max_grad']

    grad_sum_next = tree_map(
        lambda s, g: s * decay + g,
        grad_sum,
        grad
    )

    grad_squared_sum_next = tree_map(
        lambda s, g: s * decay**2 + g**2,
        grad_squared_sum,
        grad
    )

    max_grad_next = tree_map(
        lambda s, g: jnp.maximum(s * decay, jnp.abs(g)),
        max_grad,
        grad
    )

    prediction_next = tree_map(
        lambda g, m, s: -base_lr * m / jnp.sqrt(1e-8 + s) * jnp.sign(g) * (jnp.exp(eta * jnp.abs(g) / jnp.sqrt(1e-8 + s)) - 1.0),
        grad_sum_next,
        max_grad_next,
        grad_squared_sum_next,
    )

    opt_state_next = {
        'grad_squared_sum': grad_squared_sum_next,
        'prediction': prediction_next,
        'decay': decay,
        'eta': eta,
        'base_lr': base_lr,
        'grad_sum': grad_sum_next,
        'max_grad': max_grad_next
    }

    return opt_state_next, {
        'base_lr': base_lr,
    }



def simple_pistol_sq_init(params, base_lr=1.0, eta=1e-6, decay=1.0):
    state = {
        'grad_squared_sum': zeros_like(params),
        'prediction': zeros_like(params),
        'decay': decay,
        'eta': eta,
        'base_lr': base_lr,
        'grad_sum': zeros_like(params),
        'max_grad': zeros_like(params),
    }

    return state

def simple_pistol_sq_reset(old_state, epoch_count, do_decrease=True, reset_scaling=1.0):
    state = {
        'grad_squared_sum': zeros_like(old_state['grad_squared_sum']),
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'eta': old_state['eta'],
        'base_lr':  reset_scaling * old_state['base_lr']*(epoch_count + 1)/(epoch_count + 2) if do_decrease else reset_scaling * old_state['base_lr'],
        'grad_sum': zeros_like(old_state['grad_sum']),
        'max_grad': zeros_like(old_state['max_grad'])
    }

    return state   

def simple_pistol_sq_update(grad, opt_state, do_logging=False):

    grad_squared_sum = opt_state['grad_squared_sum']
    prediction = opt_state['prediction']
    decay = opt_state['decay']
    eta = opt_state['eta']
    base_lr = opt_state['base_lr']
    grad_sum = opt_state['grad_sum']
    max_grad = opt_state['max_grad']

    grad_sum_next = tree_map(
        lambda s, g: s * decay + g,
        grad_sum,
        grad
    )

    grad_squared_sum_next = tree_map(
        lambda s, g: s * decay**2 + g**2,
        grad_squared_sum,
        grad
    )

    max_grad_next = tree_map(
        lambda s, g: jnp.maximum(s * decay, jnp.abs(g)),
        max_grad,
        grad
    )

    prediction_next = tree_map(
        lambda g, m, s: -base_lr * g / (1e-8 + s)  * (jnp.exp(eta * (g**2) / (1e-8 + s)) - 1.0),
        grad_sum_next,
        max_grad_next,
        grad_squared_sum_next,
    )

    opt_state_next = {
        'grad_squared_sum': grad_squared_sum_next,
        'prediction': prediction_next,
        'decay': decay,
        'eta': eta,
        'base_lr': base_lr,
        'grad_sum': grad_sum_next,
        'max_grad': max_grad_next
    }

    return opt_state_next, {
        'base_lr': base_lr,
    }






def simple_fr_noconst_init(params, base_lr=1.0, eta=1e-6, decay=1.0):
    state = {
        'grad_squared_sum': zeros_like(params),
        'prediction': zeros_like(params),
        'decay': decay,
        'eta': eta,
        'base_lr': base_lr,
        'grad_sum': zeros_like(params)
    }

    return state

def simple_fr_noconst_reset(old_state, epoch_count, do_decrease=True, reset_scaling=1.0):
    state = {
        'grad_squared_sum': zeros_like(old_state['grad_squared_sum']),
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'eta': old_state['eta'],
        'base_lr':  reset_scaling * old_state['base_lr']*(epoch_count + 1)/(epoch_count + 2) if do_decrease else reset_scaling * old_state['base_lr'],
        'grad_sum': zeros_like(old_state['grad_sum'])
    }

    return state   

def simple_fr_noconst_update(grad, opt_state, do_logging=False):

    grad_squared_sum = opt_state['grad_squared_sum']
    prediction = opt_state['prediction']
    decay = opt_state['decay']
    eta = opt_state['eta']
    base_lr = opt_state['base_lr']
    grad_sum = opt_state['grad_sum']

    grad_sum_next = tree_map(
        lambda s, g: s * decay + g,
        grad_sum,
        grad
    )

    grad_squared_sum_next = tree_map(
        lambda s, g: s * decay**2 + g**2,
        grad_squared_sum,
        grad
    )


    prediction_next = tree_map(
        lambda g, s: -base_lr  * jnp.sign(g) * (jnp.exp(eta * jnp.abs(g) / jnp.sqrt(1e-8 + s)) - 1.0),
        grad_sum_next,
        grad_squared_sum_next,
    )

    opt_state_next = {
        'grad_squared_sum': grad_squared_sum_next,
        'prediction': prediction_next,
        'decay': decay,
        'eta': eta,
        'base_lr': base_lr,
        'grad_sum': grad_sum_next
    }

    return opt_state_next, {
        'base_lr': base_lr,
    }




def simple_fr_optimistic_init(params, base_lr=1.0, eta=1e-6, decay=1.0):
    state = {
        'normal_state': simple_fr_init(params, base_lr, eta, decay),
        'hint_state': simple_fr_init(params, base_lr, eta, decay),
        'prediction': zeros_like(params),
        'hint': zeros_like(params),
    }

    return state

def simple_fr_optimistic_reset(old_state, epoch_count, do_decrease=True, reset_scaling=1.0):
    state = {
        'prediction': zeros_like(old_state['prediction']),
        'normal_state': simple_fr_reset(old_state['normal_state'], epoch_count, do_decrease, reset_scaling),
        'hint_state': simple_fr_reset(old_state['hint_state'],  epoch_count, do_decrease, reset_scaling),
        'hint': zeros_like(old_state['hint']),
    }

    return state   

def simple_fr_optimistic_update(grad, opt_state, do_logging=False):

    prediction = opt_state['prediction']
    normal_state = opt_state['normal_state']
    hint_state = opt_state['hint_state']
    hint = opt_state['hint']

    normal_state_next, normal_logs = simple_fr_update(grad, normal_state, do_logging)

    hint_grad = tree_map(
        lambda h, g: g*h,
        grad,
        hint,
    )

    hint_next = grad

    hint_state_next, hint_logs = simple_fr_update(hint_grad, hint_state, do_logging)

    prediction_next = tree_map(
        lambda np, hp, hn: np + hp*hn, 
        normal_state_next['prediction'],
        hint_state_next['prediction'],
        hint_next,
    )


    opt_state_next = {
        'prediction': prediction_next,
        'normal_state': normal_state_next,
        'hint_state': hint_state_next,
        'hint': hint_next,
    }

    return opt_state_next, {
        'base_lr': normal_state_next['base_lr']
    }


    


def adagrad_scaled_init(params, lr=1.0, eps=1e-8, decay=1.0, base_lr=1e-5, reg=1.0):
    state = {
        'grad_squared_sum': zeros_like(params),
        'prediction': zeros_like(params),
        'decay': decay,
        'eps': eps,
        'lr': lr,
        'base_lr': base_lr,
        'reg': reg
    }

    return state

def adagrad_scaled_reset(old_state, epoch_count, do_decrease=True, reset_scaling=1.0):
    state = {
        'grad_squared_sum': zeros_like(old_state['grad_squared_sum']),
        'prediction': zeros_like(old_state['prediction']),
        'decay': old_state['decay'],
        'eps': old_state['eps'],
        'lr': old_state['lr'],
        'base_lr': reset_scaling * old_state['base_lr']*(epoch_count + 1)/(epoch_count + 2) if do_decrease else reset_scaling * old_state['base_lr'],
        'reg': old_state['reg']
    }

    return state    

def adagrad_scaled_update(grad, opt_state, do_logging=False):

    grad_squared_sum = opt_state['grad_squared_sum']
    prediction = opt_state['prediction']
    decay = opt_state['decay']
    eps = opt_state['eps']
    lr = opt_state['lr']
    base_lr = opt_state['base_lr']
    reg = opt_state['reg']

    grad_squared_sum_next = tree_map(
        lambda s, g: s * decay + g**2,
        grad_squared_sum,
        grad
    )

    prediction_next = tree_map(
        lambda p, g, s: p * decay \
            - lr * (jnp.abs(p) + base_lr) * (g / (eps + jnp.sqrt(s)) + reg * jnp.sign(p) * g**2/(eps + s)),
        prediction,
        grad,
        grad_squared_sum_next
    )

    opt_state_next = {
        'grad_squared_sum': grad_squared_sum_next,
        'prediction': prediction_next,
        'decay': decay,
        'eps': eps,
        'lr': lr,
        'base_lr': base_lr,
        'reg': reg,
    }

    return opt_state_next, {
        'base_lr': base_lr,
    }




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


def cb_stable_reset(old_state, count, do_decrease=True, reset_scaling=1.0):
    state = {
        'wealth': tree_map( lambda x: jnp.full_like(x, fill_value=old_state['eps']/(count+1)), old_state['wealth']),
        'bet_fractions': zeros_like(old_state['bet_fractions']),
        'subepoch_grad': zeros_like(old_state['subepoch_grad']),
        'max_grads': old_state['max_grads'],
        'subepoch_grad_abs_sum': zeros_like(old_state['subepoch_grad_abs_sum']),
        'subepoch_grad_sum': zeros_like(old_state['subepoch_grad_sum']),
        'eta': old_state['eta'],
        'eps': reset_scaling*old_state['eps']*(count + 1)/(count + 2) if do_decrease else reset_scaling * old_state['eps'],
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
            Note that 'none' is the string 'none', not the python value None.

        use_loss_diff: if true, we will rescale the gradient so that
            <gradient , update> = loss(param + update) - loss(param)
            
            **** NOTE: if you set use_loss_diff=true, you probably want to set
            rand_scaling_type = 'none'


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
        'total_reward': jnp.zeros(1),
        'prev_rand_scaling': jnp.zeros(1),
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
    prev_rand_scaling = opt_state['prev_rand_scaling']
    total_reward = opt_state['total_reward']

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

    total_reward_next = total_reward + prev_lr * ol_grad

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



    if 'uniform' in rand_scaling_type:
        cur_rng, rng  = jax.random.split(rng)
        rand_scaling = jax.random.uniform(cur_rng)
        rand_scaled_lr = rand_scaling * final_lr
        true_scaled_lr = final_lr
    elif 'exponential' in rand_scaling_type:
        cur_rng, rng  = jax.random.split(rng)
        rand_scaling = -jnp.log(jax.random.uniform(cur_rng))
        rand_scaled_lr = rand_scaling * final_lr
        true_scaled_lr = rand_scaling * final_lr # the point of exponential is that these are the same!
    elif 'half' in rand_scaling_type:
        rand_scaled_lr = 0.5 * final_lr
        true_scaled_lr = final_lr
    elif 'none' in rand_scaling_type:        
        rand_scaled_lr = final_lr
        true_scaled_lr = final_lr
    else:
        raise(NotImplementedError, "unknown rand scaling!")

    # clip gradients
    clipped_grad = clip_by_variable_norm(grad, clip)
    update, optax_state_next = optax_opt.update(clipped_grad,  optax_state, params)

    if 'scale_by_lr' in rand_scaling_type:
        update = tree_map(
            lambda u: lr * u,
            update
        )


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
        'prev_lr': final_lr,
        'prev_rand_scaling': rand_scaled_lr,
        'total_reward': total_reward_next,
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
            'ol_optax_total_reward': total_reward_next,
            'grad_norm': tree_norm(grad),
            'scaled_update_norm': rand_scaled_lr * tree_norm(update),
            'ol_grad': ol_grad,
            'update_norm': tree_norm(update),
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



def OL_momentum_init(params,
                     ol_init,
                     ol_args,
                     ol_kwargs,
                     ol_update_fn,
                     ol_reset_fn,
                     ol_reset_kwargs={'do_decrease': False},
                     ol_update_kwargs={},
                     weight_decay=0.0,
                     reset_threshold=100.0,
                     rand_scaling_type='uniform'):
    '''
    ol_momentum_init: optimization by reduction to online learning

        params: model parameters
        ol_init: initialization function for online learner
        ol_args: argument list for ol_init
        ol_kwargs: keyword argument dictionary for ol_init
        ol_update_fn: update function for online learner
        ol_reset_fn: function that resets the state of the online learner
        ol_reset_kwargs: static kwargs to provide to ol_reset_fn.
        ol_update_kwargs: static kwargs to provde to ol_update_fn.
        reset_threshold: reset the online learner when the reward exceeds this
            value.
        rand_scaling_type: how to scale the online learner's iterates.
            should be a string, currently supports:
            'uniform' and 'half'
    returns:
        state: state of the optimizer.
        update_fn: function to use for applying the optimizer to a model.
    '''
    state = {
        'ol_state': ol_init(params, *ol_args, **ol_kwargs),
        'true_params': params,
        'last_offset': zeros_like(params),
        'total_reward': jnp.zeros(1),
        'epoch_reward': jnp.zeros(1),
        'max_epoch_reward': jnp.zeros(1),
        'last_constraint': 1.0,
        'epoch_count': 0,
        'bad_epoch_count': 0,
        'iteration_count':  0,
        'reset_threshold': reset_threshold,
        'epoch_start_true_params': params,
        'weight_decay': weight_decay,
    }

    return state, functools.partial(
        OL_momentum_update,
        rand_scaling_type,
        ol_update_fn,
        functools.partial(ol_reset_fn, **ol_reset_kwargs),
        **ol_update_kwargs
        )

def OL_momentum_update(rand_scaling_type,
                       ol_update,
                       ol_reset,
                       loss_and_grad_fn,
                       rng,
                       model_and_example_state,
                       opt_state,
                       scale=jnp.array(1.0),
                       do_logging=False,
                       reset_scaling=0.5,
                       do_decrease_guard=False,
                       guard_threshold=0.01,
                       constraint_type='infinity'):
    '''
    update function for online learning based optimizer.
    
    rand_scaling_type: string, describes how to scale online learner.
        can be 'uniform' or 'half' or 'none'.
    ol_update: function for doing the online learning update
    ol_reset: function for resetting the online learner.

    rng <-> do_logging: see description for optax_ol_scaled_update.

    reset_scaling: amount to scale ol "lr" analog by when doing a reset triggered
        by do_decrease_guard.
    do_decrease_guard: if true, check if there is a reward decrease (indicating
        that the loss is increasing), and if so reset to before the decrease happened
        and also decrease the model's learning rate parameter by a factor of reset_scaling.
    guard_threshold: if do_decrease_guard == True, then every iteration checks if the total reward
        has decreased by this guard_threshold from the maximum value. If so, we decrease the learning
        rate and reset to the beginning of the "epoch".
    constraint_type: string describing how we enforce the constraints specified by "scale" argument (basically,
        the learning rate). If 'infinity', then we project updates into an L_infinity ball (i.e. clip the 
        coordinates to [-scale, scale]). If 'l2', then we project into an l2 ball.
        if 'per_layer_l2', then we project each *layer* (i.e. leaf of the JAX pytree of updates) into an
        L2 ball.

        if constraint_type == 'adaptive', then we use L2 constraints, but the use of scaling is different.
        In this mode, we set the constraint to be: (scaling + max(total_reward, 0))/( (1e-8 + grad_norm) * sqrt(iteration_count))

        Not a lot of theory behind this, but here is the intuition:
        If we were making truly random-walk progress, we would expect to make about Lipschitz constant * constraint * sqrt(iterations)
        change in the loss. So, let us at least constraint ourselves to ensure that this value is no larger than
        our actual progress. This should be an over-estimate since actually we are presumably not a random walk.
    
    '''

    value, grad = loss_and_grad_fn(model_and_example_state)
    grad_norm = tree_norm(grad)
    model_state = model_and_example_state['model_state']

    cur_rng, rng = jax.random.split(rng)

    if rand_scaling_type == 'uniform':
        rand_scaling = jax.random.uniform(cur_rng)
        true_scaling = 1.0
    elif rand_scaling_type == 'half':
        rand_scaling = 0.5
        true_scaling = 1.0
    elif rand_scaling_type == 'none':
        rand_scaling = 1.0
        true_scaling = 1.0
    elif rand_scaling_type == 'exponential':
        rand_scaling = jax.random.exponential(cur_rng)
        true_scaling = rand_scaling
 
    constants = model_state['constants']
    params = model_state['params']

    ol_state = opt_state['ol_state']
    true_params = opt_state['true_params']
    total_reward = opt_state['total_reward']
    epoch_reward = opt_state['epoch_reward']
    last_offset = opt_state['last_offset']
    epoch_count = opt_state['epoch_count']
    iteration_count = opt_state['iteration_count']
    reset_threshold = opt_state['reset_threshold']
    epoch_start_true_params = opt_state['epoch_start_true_params']
    bad_epoch_count = opt_state['bad_epoch_count']
    max_epoch_reward = opt_state['max_epoch_reward']
    weight_decay = opt_state['weight_decay']
    last_constraint = opt_state['last_constraint']

    # grad = tree_map(
    #     lambda g, p: g + weight_decay * p,
    #     grad,
    #     params
    # )

    iteration_count_next = iteration_count + 1

    ol_prediction = ol_state['prediction']

    reward_increment = tree_dot(last_offset, grad)
    
    # tree_reduce(
    #     lambda s, x: s+x,
    #     tree_map(
    #         lambda o, g: jnp.sum(o*g),
    #         last_offset,
    #         grad
    #     )
    # )

    total_reward_next = total_reward - reward_increment
    epoch_reward_next = epoch_reward - reward_increment

    max_epoch_reward_next = jnp.maximum(max_epoch_reward, epoch_reward_next)

    temp_epoch_reward_next = epoch_reward_next


    # apply constraint set reduction.
    if constraint_type == 'infinity':
        constrained_grad = tree_map(
            lambda o, g: jnp.where(-o * jnp.sign(g) >= last_constraint, jnp.zeros_like(g), g),
            ol_prediction,
            grad)
    elif constraint_type == 'l2' or constraint_type == 'adaptive':
        prev_prediction_norm = tree_norm(ol_prediction)

        # compute gradient projection along prediction direction. We only need this if it is negative, and the prediction norm
        # is bigger than the constraint. That is, we only want to correct the gradient if the gradient is
        # pushing us further outside the constraint.
        prediction_grad_projection = jnp.minimum(tree_dot(ol_prediction, grad)/prev_prediction_norm**2, 0) * (prev_prediction_norm > last_constraint)
        constrained_grad = tree_map(
            lambda p, g: g - prediction_grad_projection * p,
            ol_prediction,
            grad
        )
    elif constraint_type == 'per_layer_l2':
        prev_prediction_layer_norms = tree_map(
            lambda x: 1e-8 + jnp.linalg.norm(x),
            ol_prediction
        )

        # compute gradient projection along prediction direction for each layer. We only need this if it is negative, and the prediction norm
        # is bigger than the constraint. That is, we only want to correct the gradient if the gradient is
        # pushing us further outside the constraint.
        prediction_grad_projection = tree_map(
            lambda olp, g, norm: jnp.minimum(
                jnp.sum( olp * g )/norm**2, 0
            ) * (norm > last_constraint),
            ol_prediction,
            grad,
            prev_prediction_layer_norms
        )

        constrained_grad = tree_map(
            lambda p, g, pgp: g - pgp * p,
            ol_prediction,
            grad,
            prediction_grad_projection,
        )



    ol_state_next, ol_logs = ol_update(constrained_grad, ol_state, do_logging)

    ol_state_next, epoch_reward_next, max_epoch_reward_next, epoch_count_next, epoch_start_true_params_next = jax.lax.cond(
        jnp.all(epoch_reward_next > reset_threshold),
        lambda s, r, mr, c, esp, tp: (ol_reset(s, c), jnp.zeros_like(r), jnp.zeros_like(mr),  c + 1, tp),
        lambda s, r, mr, c, esp, tp: (s, r, mr, c, esp),
        ol_state_next, epoch_reward_next, max_epoch_reward_next, epoch_count, epoch_start_true_params, true_params)


    # apply constraints
    if constraint_type == 'infinity':
        offset = tree_map(
            lambda ol_p, p: jnp.clip(ol_p, a_min=-scale, a_max=scale) - scale * weight_decay * p,
            ol_state_next['prediction'],
            true_params,
        )
        # last_constraint_update = scale
    elif constraint_type == 'l2' or constraint_type == 'adaptive':

        if constraint_type == 'adaptive':
            scale = (scale + jnp.maximum(total_reward_next, 0))/((1e-8  + grad_norm) * jnp.sqrt(iteration_count_next))

        ol_prediction_norm = tree_norm(ol_state_next['prediction'])
        truncated_scale = jnp.minimum(1.0, scale/ol_prediction_norm)

        offset = tree_map(
            lambda ol_p, p: ol_p * truncated_scale - scale * weight_decay * p,
            ol_state_next['prediction'],
            true_params,
        )
        # last_constraint_update = scale
    elif constraint_type == 'per_layer_l2':
        truncated_scale = tree_map(
            lambda ol_p: jnp.minimum(1.0, scale/(1e-8 + jnp.linalg.norm(ol_p))),
            ol_state_next['prediction']
        )
        offset = tree_map(
            lambda ol_p, p, trunc: ol_p * trunc - scale * weight_decay * p,
            ol_state_next['prediction'],
            true_params,
            truncated_scale,
        )

    last_constraint_update = scale

    params_next = tree_map(lambda p, o: p+ rand_scaling * o, true_params, offset)

    if do_decrease_guard:
        params_next, ol_state_next, epoch_reward_next, max_epoch_reward_next, total_reward_next, offset, bad_epoch_count_next = jax.lax.cond(
            jnp.all(epoch_reward_next > max_epoch_reward_next - guard_threshold),
            lambda p, s, ps, c, en, me, trn, o, bc: (p, s, en, me, trn, o, bc),
            lambda p, s, ps, c, en, me, trn, o, bc: (
                ps,
                ol_reset(s, c, reset_scaling=reset_scaling),
                jnp.zeros_like(en),
                jnp.zeros_like(me),
                trn - en,
                zeros_like(o),
                bc+1),
            params_next,
            ol_state_next,
            epoch_start_true_params_next,
            epoch_count_next,
            epoch_reward_next,
            max_epoch_reward_next,
            total_reward_next,
            offset,
            bad_epoch_count,
        )
    else:
        bad_epoch_count_next = bad_epoch_count

    model_state_next = {
        'constants': constants,
        'params': params_next
    }

    true_params_next = tree_map(lambda p, o: p+ true_scaling*o, true_params, offset)

    opt_state_next = {
        'ol_state': ol_state_next,
        'true_params': true_params_next,
        'total_reward': total_reward_next,
        'epoch_reward': epoch_reward_next,
        'iteration_count': iteration_count_next,
        'epoch_count': epoch_count_next,
        'reset_threshold': reset_threshold,
        'last_offset': offset,
        'epoch_start_true_params': epoch_start_true_params_next,
        'bad_epoch_count': bad_epoch_count_next,
        'max_epoch_reward': max_epoch_reward_next,
        'weight_decay': weight_decay,
        'last_constraint': last_constraint_update,
    }


    if do_logging:
        if constraint_type == 'infinity':
            max_update_norm = scale * tree_norm(ones_like(offset))
        else:
            max_update_norm = scale
        log_dict = {
            'total_reward': total_reward_next,
            'epoch_reward': epoch_reward_next,
            'epoch_count': epoch_count_next,
            'bad_epoch_count': bad_epoch_count_next,
            'temp_epoch_reward': temp_epoch_reward_next,
            'reward_increment': reward_increment,
            'max_epoch_reward': max_epoch_reward_next,
            'update_norm': tree_norm(offset),
            'unconstrained_update_norm': tree_norm(ol_state_next['prediction']),
            'unconstrainted_update_diff': tree_norm(tree_map(
                lambda x, y: x-y,
                ol_state_next['prediction'],
                ol_state['prediction']
            )),
            'grad_norm': grad_norm,
            'max_update_norm': max_update_norm,
            'weight_decay_penalty': weight_decay * tree_norm(params)**2,
            'param_norm': tree_norm(params),
            'const_grad_pred_prod': tree_dot(constrained_grad, ol_prediction),
        }
        if constraint_type == 'per_layer_l2':
            log_dict['max_per_layer_update_norms'] = tree_map(
                lambda x: jnp.linalg.norm(x),
                offset
            )
        log_dict.update(ol_logs)
        # for key, value in ol_logs.items():
        #     log_dict['ol_'+key] = value
    else:
        log_dict = None


    return rng, value, grad, model_state_next, opt_state_next, log_dict

    


####### more optax-compatible version ######

# usage example (assuming simple_fr_init is the online learner. It takes two main arguments: base_lr and eta.
# eta should possibly not be bigger than 1/sqrt(2)~0.7 unless you want the scale to grow even with random inputs.
# however, I'm not sure that growing with random inputs is actually bad...)
#
# # CAUTION: be careful clipping the "grad" argument you provide to learned_scale_update: it has been
# # more sensitive to aggressive clipping in some experiments.
#
# initialization:
#
# params = model_state['params'] # or however the params are stored.
# rescaling_state = init_learned_scale(params, simple_fr_init, base_lr=1e0, eta=1.0, decay=0.999, min_bound=1e-6, max_bound=1e1)
# rescaling_fn = functools.partial(learned_scale_update, simple_fr_update, rand_scale_type='uniform', residual=True)
#
# (probably initialize your optax uptimizer with whatever standard learning rate schedule or similar).
# (alternatively, you could initialize the optax optimizer with a larger constant learning rate, but
#  for numerical precision issues it might be important to be not too big. Technically, the "first 
#  learning rate" that the learner will try will be ~ base_lr * optax_lr, and theoretically if
#  this product is held constant at initialization the computations will be identical. However
#  in practice I think having a large optax_lr can cause some issues with adding small and large
#  numbers together.)
#
# # in training loop:
# grad = grad_fn(params, minibatch)
#
# updates, opt_state = optax_optimizer.update(grad,  optax_state, model_state['params'])
# ### note: updates could (and maybe should) already have some learning rate schedule applied!
# ### then, the scaling will learn a way to rescale that schedule appropriately.
#
# scaling_state, rng, rescaled_updates = rescaling_fn(rng, rescaling_state, grad, updates)
# params = optax.apply_updates(params, rescaled_updates)
#


def init_learned_scale(params, ol_init=simple_fr_init, min_scale=1e-6, max_scale=1e1, *ol_args, **ol_kwargs):
    ol_state = ol_init(jnp.zeros(1), *ol_args, **ol_kwargs)
    update_state = {
        'prev_updates': zeros_like(params),
        'missing_scaling': jnp.zeros(1),
        'ol_state': ol_state,
        'min_scale': min_scale,
        'max_scale': max_scale,
        'constraint_violation': jnp.zeros(1),
        'estimated_loss_gap': jnp.zeros(1),
        'prev_nonrand_scaling': jnp.zeros(1),
    }

    return update_state

def learned_scale_update(ol_update, rng, state, grad, updates, rand_scale_type, residual,):
    '''
    rescales an update with an online learner.
    
    args:
        ol_update: function that updates an online learner.
            Takes as input an online learner state and a (meta)-gradient.
            Returns a new online learner state and some logs dict (which we ignore).
            *** this argument is not a jax type, so it should be removed with a partial
                or set as a static argument in order to compile ***

        rng: jax rng.
        state: state for this scaling learner.
        grad: gradient of loss.
        update: update from base algorithm to rescale. Should be same
            pytree shape as grad.

        rand_scale_type: 'uniform', 'exponential', or 'none'
        residual: boolean flag indicating whether to learn a residual or not.
        *** these arguments are also not JAX types...***

    returns:
        state_next: next value for the state.
        rng: next value for the rng.
        rescaled_update: rescaled value of the update that can be applied via
            optax.apply_updates.
        log_data: a dict of values it might be useful to log somewhere (can be ignored
            if no logging is desired)
    '''


    prev_updates = state['prev_updates']
    missing_scaling = state['missing_scaling']
    ol_state = state['ol_state']
    min_scale = state['min_scale']
    max_scale = state['max_scale']
    constraint_violation = state['constraint_violation']
    estimated_loss_gap = state['estimated_loss_gap']
    prev_nonrand_scaling = state['prev_nonrand_scaling']


    rng, cur_rng = jax.random.split(rng)

    if rand_scale_type == 'uniform':
        rand = jax.random.uniform(cur_rng)
        missing_rand = (1.0 - rand)
    elif rand_scale_type == 'exponential':
        rand = jax.random.exponential(cur_rng)
        missing_rand = 0.0
    elif rand_scale_type == 'half':
        rand = 0.5
        missing_rand = 0.5
    elif rand_scale_type == 'none':
        rand = 1.0
        missing_rand = 0.0
    elif rand_scale_type == 'debug_rand':
        rand = jax.random.uniform(cur_rng)
        missing_rand = 0.0

    if residual:
        base_scale_value = 1.0
    else:
        base_scale_value = 0.0
    

    ol_grad = tree_dot(prev_updates, grad)

    estimated_loss_gap_next = estimated_loss_gap + prev_nonrand_scaling * ol_grad

    # process gradient for enforcing constraints:
    ol_grad = jnp.where(
            -constraint_violation * jnp.sign(ol_grad) > 1e-8, # should be == 0 when no clipping happens, but idk about floating point stuff.
            jnp.zeros_like(ol_grad),
            ol_grad)
    
    ol_state_next, ol_logs = ol_update(ol_grad, ol_state)
    del ol_logs

    unclipped_nonrand_scale = base_scale_value + ol_state_next['prediction']

    clipped_nonrand_scale = jnp.clip(unclipped_nonrand_scale, a_min=min_scale, a_max=max_scale)

    constraint_violation_next = unclipped_nonrand_scale - clipped_nonrand_scale

    scaling = rand * clipped_nonrand_scale
    missing_scaling_next = missing_rand * clipped_nonrand_scale


    #
    # true_param_{t} = true_param_{t-1} + nonrand_scale_{t-1} * update_{t-1}
    # param_{t} = true_param_{t-1} + rand_{t-1} * nonrand_scale_{t-1} * update_{t-1}
    #
    # param_{t+1} = true_param_{t} + rand * nonrand_scale_{t} * update_{t}
    #             = param_{t} + (1-rand_{t-1}) * nonrand_scale_{t-1} * update_{t-1} + rand_t * nonrand_scale_{t} * update_{t}
    rescaled_update = tree_map(
        lambda u, p_u: missing_scaling * p_u + scaling * u, 
        updates,
        prev_updates,
    )

    state_next = {
        'prev_updates': updates,
        'missing_scaling': missing_scaling_next,
        'ol_state': ol_state_next,
        'min_scale': min_scale,
        'max_scale': max_scale,
        'constraint_violation': constraint_violation_next,
        'estimated_loss_gap': estimated_loss_gap_next,
        'prev_nonrand_scaling': clipped_nonrand_scale,
    }

    log_data = {
        'estimated_loss_gap': estimated_loss_gap_next,
        'random_learned_scaling': scaling,
        'nonrandom_learned_scaling': clipped_nonrand_scale,
        'ol_prediction': ol_state_next['prediction'],
        'grad_norm': tree_norm(grad),
        'update_norm': tree_norm(updates),
        'rand_value': rand,
    }

    return state_next, rng, rescaled_update, log_data






##### randomly scale the update ######
#
# These functions simply randomly scale the updates, without using an online learner.
#
# usage is nearly the same, but no need to provide online learning info.



def init_random_scale(params):
    update_state = {
        'prev_updates': zeros_like(params),
        'missing_scaling': jnp.zeros(1),
        'estimated_loss_gap': jnp.zeros(1),
    }

    return update_state


def random_scale_update(rand_scaling_type, rng, state, grad, updates):
    '''
    rescales an update with a random number.
    
    args:
        rand_scaling_type: can be 'uniform', 'exponential', or 'none'
        *** CAUTION: FIRST ARGUMENT IS NOT A JAX TYPE AND SO CANNOT BE JITTED ***

        rng: jax rng.
        state: state for this scaling learner.
        grad: (unused, only here so that it has the same signature as learned_scale_update).
        updates: updates from base algorithm to rescale. Should be same
            pytree shape as the "params" argument provided to "init_random_scale".

    
    returns:
        state_next: next value for the state.
        rng: next value for the rng.
        rescaled_update: rescaled value of the update that can be applied via
            optax.apply_updates.
        log_data: a dictionary of stuff to log 
    '''


    prev_updates = state['prev_updates']
    missing_scaling = state['missing_scaling']
    estimated_loss_gap = state['estimated_loss_gap']

    grad_product = tree_dot(prev_updates, grad)

    estimated_loss_gap_next = estimated_loss_gap + grad_product

    rng, cur_rng = jax.random.split(rng)

    if rand_scaling_type == 'uniform':
        rand = jax.random.uniform(cur_rng)
        missing_scaling_next = (1.0 - rand)
    elif rand_scaling_type == 'exponential':
        rand = jax.random.exponential(cur_rng)
        missing_scaling_next = 0.0
    elif rand_scaling_type == 'none':
        rand = 1.0
        missing_scaling_next = 0.0
    elif rand_scaling_type == 'half':
        rand = 0.5
        missing_scaling_next = 0.5
    else:
        raise ValueError(f'unknown rand_scaling_type: {rand_scaling_type}')


    #
    # true_param_{t} = true_param_{t-1} + nonrand_scale_{t-1} * update_{t-1}
    # param_{t} = true_param_{t-1} + rand_{t-1} * update_{t-1}
    #
    # param_{t+1} = true_param_{t} + rand * nonrand_scale_{t} * update_{t}
    #             = param_{t} + (1-rand_{t-1}) * update_{t-1} + rand_t * update_{t}
    rescaled_updates = tree_map(
        lambda u, p_u: missing_scaling * p_u + rand * u, 
        updates,
        prev_updates,
    )

    state_next = {
        'prev_updates': updates,
        'missing_scaling': missing_scaling_next,
        'estimated_loss_gap': estimated_loss_gap_next,
    }

    return state_next, rng, rescaled_updates, {}



##### 
# OPTIMIZER THAT IS (MOSTLY) ACTUALLY ANALYZED IN THE PAPER
#
#  usage:
#
# opt_state, update_fn = wrap_ol_momentum_like_optax(params, lr, rng, weight_decay,
#    ol_init=simple_fr_noconst_init,
#    ol_update_fn=simple_fr_noconst_update,
#    ol_reset_fn=simple_fr_noconst_reset,
#    ol_kwargs={
#        'base-lr':  1e-4,
#        'eta': 0.5,
#        'decay': 0.999
#    },
#    ol_update_kwargs={'constraint_type': 'l2'}, # best to leave all other arguments as default probably. These are the only ones I'd recommend trying to change.
#    rand_scaling_type='none', # this one controls whether randomization is used and if so what type. Set to 'uniform' for uniform in [0,1]. set to 'none' to turn off randomization.
#    )
#
#   the 'constraint_type' argument influences how the "lr" parameter is interpreted:
#   if it is 'l2', then the "lr" is a bound on the l2 norm of the update. If it is 'infinity', then "lr" is a bound on the infinity norm of the update.
#   if it is 'per_layer_l2' then "lr" is a bound on the norm of each layer's update.
#
#
# in training loop:
# grads = grad_fn(params, minibatch)  # should have same pytree shape  as params argument above.
#
# opt_state, updates, logs = update_fn(grad, opt_state, params) # could optionally provide the do_logging flag as a final argument here if desired.
# params = optax.apply_updates(params, updates)
#
#
# if you want to adjust the learning rate on-the-fly, it is accesible in opt_state['lr']
# you can also now provide a fourth "lr_override" argument to update_fn, which will multiply the opt_state['lr'] by the
# provided argument during that step.
#
#####
    


def wrap_ol_momentum_like_optax(
    params,
    lr,
    rng,
    weight_decay=0.0,
    ol_init=simple_fr_noconst_init,
    ol_args=[],
    ol_kwargs={
        'base-lr':  1e-4,
        'eta': 1.0,
        'decay': 0.99
    },
    ol_update_fn=simple_fr_noconst_update,
    ol_reset_fn=simple_fr_noconst_reset,
    ol_reset_kwargs={'do_decrease': False}, # this will never be called anyway by default unless 'reset_threshold' is set to a non-huge value.
    ol_update_kwargs={'constraint_type': 'l2'},
    reset_threshold=1e10, #just turn this off by default,
    rand_scaling_type='uniform',
    do_logging=True):
    '''
    arguments:
        params:  model parameters pytree
        lr: "learning rate" (actually a constraint on the update norm)
        rng: jax rng (you should already have split the rng before providing it here)
        weight_decay: weight decay (still not sure the right way to apply this, so it may or may not be a great idea).
        
        ol_init: initialization function for online learner
        ol_args: arguments  to provide to online learner
        ol_kwargs: kwarguments for online learning
        ol_reset_fn: reset function (recommend that this be not used by setting 'reset_threshold' really large later. It is a bit experimental.
        ol_update_kwargs: extra kwargs for ol_momentum_update. Probably should not override the do_decrease_guard=False default (it is also experimental)
            but constraint_type could be either 'l2' or 'infinity' to indicate whether the "lr" represents a per-coordinate or global constraint.
            Can also set constraint_type to 'per_layer_lr' to do l2 constraint layerwise (sort of a lamb-like mix of l2/infinity)
        reset_threshold: make this large.
        rand_scaling_type: could be 'uniform', 'exponential' or 'none'.
        do_logging: whether to output logs.


    returns:
        state: optimizer state pytree.
        update_fn: a function that performs the updates. It is nearly the same as optax optimizer:
            update_fn(grad, state, params) will return
            updates, next_state, log_dict
            and updates can be applied with optax.apply_updates.
            log_dict is a dictionary of logging info. 
    '''


    ol_momentum_state, update_fn = OL_momentum_init(
        params,
        ol_init,
        ol_args,
        ol_kwargs,
        ol_update_fn,
        ol_reset_fn,
        ol_reset_kwargs,
        ol_update_kwargs,
        weight_decay,
        reset_threshold,
        rand_scaling_type
    )

    state = {
        'rng': rng,
        'lr': lr,
        'ol_momentum_state': ol_momentum_state
    }

    def get_updates(grad, state, params, lr_rescale=1.0, do_logging=do_logging):

        # we just lie about the loss. The optimizer never looks at it anyway.
        loss_and_grad_fn = lambda x: ({'loss': 0.0}, grad)

        # lie about the constants in the model state also:
        model_state = {
            'params': params,
            'constants': jnp.zeros(1),
        }

        model_and_example_state = {
            'model_state': model_state
        }


        rng = state['rng']
        scale = state['lr']
        ol_momentum_state = state['ol_momentum_state']


        

        rng_next, value, grad, model_state_next, opt_state_next, log_dict = update_fn(
            loss_and_grad_fn,
            rng,
            model_and_example_state,
            ol_momentum_state,
            scale * lr_rescale,
            do_logging=do_logging
        )

        state_next = {
            'rng': rng_next,
            'lr': scale,
            'ol_momentum_state': opt_state_next,
        }

        params_next = model_state_next['params']

        updates  = tree_map(
            lambda a, b: a - b,
            params_next,
            params
        )

        return updates, state_next, log_dict

    return state, get_updates


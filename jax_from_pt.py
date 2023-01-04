


import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map

import pprint


from types import SimpleNamespace

import functools

import torch

# t_lin = torch.nn.Linear(10, 20)

# w = jnp.array(t_lin.weight.detach().numpy())
# b = jnp.array(t_lin.bias.detach().numpy())

# t_x = torch.normal(torch.zeros(10))

# x = jnp.array(t_x.numpy())

# print(f"t_x: {t_x}\nlin(x): {t_lin(t_x)}")

# print(f" w shape: {w.shape} b shape: {b.shape}")

# print(f"{b + jnp.matmul(w, x)}")

class Identity:
    def init():
        t_module = torch.nn.Identity()
        state = {
            'params': {},
            'constants': {}
        }
        t_state = {
            'params': {},
            'constants': {}
        }
    
        return state, Identity.apply, t_module, t_state
        
    def apply(state, x):
        return x


class Linear:
    # don't instantiate this class, it's just for static methods
    def init(in_features, out_features, bias=True):

        t_lin = torch.nn.Linear(in_features, out_features, bias)

        w = jnp.array(t_lin.weight.detach().numpy())
        params = {
            'weight': w
        }
        t_params = {
            'weight': t_lin.weight
        }
        if bias:
            b = jnp.array(t_lin.bias.detach().numpy())
            params['bias'] = b
            t_params['bias'] = t_lin.bias

        return {
            'params': params,
            'constants': {}
            }, Linear.apply, t_lin, {
                'params': t_params,
                'constants': {}
            }

    def apply(state, input):
        # print("tracing")
        params = state['params']
        weight = params['weight'].transpose()


        r = jnp.matmul(input, weight)

        if 'bias' in params:
            bias = params['bias']
            r = r + bias

        return r


class Embedding:

    def init(num_embeddings, embedding_dim):

        t_embed = torch.nn.Embedding(num_embeddings, embedding_dim)

        weight = jnp.array(t_embed.weight.detach().numpy())

        return {
            'params': {
                'weight': weight
            },
            'constants': {}
        }, Embedding.apply, t_embed, {
            'params': {
                'weight': t_embed.weight
            },
            'constants': {}
        }

    def apply(state, idx):
        weight = state['params']['weight']
        return weight[idx]

def repack_list(states):
    params = [s['params'] for s in states]
    constants = [s['constants'] for s in states]

    return {
        'params': params,
        'constants': constants
    }

class Sequential:

    def init(*states_and_applies):
        states = [s_a[0] for s_a in states_and_applies]
        applies = [s_a[1] for s_a in states_and_applies]
        t_modules = [s_a[2] for s_a in states_and_applies]
        t_params = [s_a[3] for s_a in states_and_applies]

        seq_state = repack_list(states)

        seq_t_params = repack_list(t_params)

        apply_fn = functools.partial(Sequential.apply, applies)

        return seq_state, apply_fn, torch.nn.Sequential(*t_modules), seq_t_params

    
    def apply(applies, state, x):
        states = [
            {
                'params': p,
                'constants': c
            }
            for p, c in zip(state['params'], state['constants'])
        ]

        for s, f in zip(states, applies):
            x = f(s, x)

        return x


class Module:

    def __init__(self):
        self._own_params = {}
        self._own_constants = {}

        self._t_own_params = {}
        self._t_own_constants = {}

        self._apply_fns = {}
        self._t_params = {}
        self._t_modules = {}

        self._sub_modules = {}

        self.config = SimpleNamespace()


    def __setattr__(self, name, value):
        if name == 'config' or name[0] == '_': # reserve names starting with _ to be assigned as normal.
            return super().__setattr__(name, value)
        # look, for now let's assume all attributes are also modules ok? thanks.

        assert name not in self._own_params, f"cannot create submodule {name}: a pre-existing parameter already has this name!"
        assert name not in self._own_constants, f"cannot create submodule {name}: a pre-existing constant already has this name!"

        state, apply, t_module, t_params = value



        self._sub_modules[name] = state
        self._apply_fns[name] = apply
        self._t_modules[name] = t_module
        self._t_params[name] = t_params


        return super().__setattr__(name, value)

    def get_state(self):
        params = {}
        constants = {}
        for name, value in self._sub_modules.items():
            params[name] = value['params']
            constants[name] = value['constants']

        params.update(self._own_params)
        constants.update(self._own_constants)

        return {
            'params': params,
            'constants': constants
        }

    def get_t_state(self):
        params = {}
        constants = {}
        for name, value in self._t_params.items():
            params[name] = value['params']
            constants[name] = value['constants']

        params.update(self._t_own_params)
        constants.update(self._t_own_constants)

        return {
            'params': params,
            'constants': constants
        }

    def get_all_t_params(self):
        ret = {}
        for name, value in self._t_modules.items():
            ret[name] = value
        ret.update(self._t_own_constants)
        ret.update(self._t_own_params)
        ret.update({'config': self.config})

        return ret

    def setup_t_module(self, t_module):
        for name, value in self._t_modules.items():
            t_module.__setattr__(name, value)
        for name, value in self._t_own_params.items():
            t_module.register_parameter(name, torch.nn.Parameter(value))
        for name, value in self._t_own_constants.items():
            t_module.register_buffer(name, value)

        t_module.config = self.config
        return t_module


    def get_own_values(self, state):
        params = {k: state['params'][k] for k in self._own_params}
        constants = {k: state['constants'][k] for k in self._own_constants}
        return {
            'params': params,
            'constants': constants
        }
    
    def __call__(self, state):
        sub_states = self.get_submodule_states(state)
        values ={
                k: functools.partial(fn, sub_states[k])
            for k, fn in self._apply_fns.items()
        }
        values.update(self._own_params)
        values.update(self._own_constants)
        values.update({'config': self.config})

        return SimpleNamespace(**values)
    

    def get_applies(self):
        return self._apply_fns
    
    def get_t_modules(self):
        return self._t_modules

    def get_submodule_states(self, state):
        return {
            k: {
                'params': state['params'][k],
                'constants': state['constants'][k]

            }
            for k in self._sub_modules
        }




    def register_buffer(self, name, value, t_value):
        assert name not in self._sub_modules, f"cannot register constant buffer {name}: a pre-existing submodule already has this name!"

        self._own_constants[name] = value
        self._t_own_constants[name] = t_value

    def register_parameter(self, name, value, t_value):
        assert name not in self._sub_modules, f"cannot register parameter {name}: a pre-existing submodule already has this name!"

        self._own_params[name] = value
        self._t_own_params[name] = t_value



class TModule(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        module.setup_t_module(self)




class MyModule:
    def init(vocab, embed, dim1, dim2, dim3=1):

        module = Module()

        module.embed = Embedding.init(vocab, embed)
        module.seq = Sequential.init(Linear.init(embed, dim1), Linear.init(dim1, dim2))

        t_mul = torch.normal(torch.ones(dim2))
        j_mul = t_mul.numpy()
        module.register_buffer('mul', j_mul, t_mul)

        module.fc2 = Linear.init(dim2, dim3)


        state = module.get_state()
        t_state = module.get_t_state()

        apply = functools.partial(MyModule.apply, module)


        return state, apply, MyTModule(module), t_state

    def apply(module, state, x):

        module = module(state)

        x = module.embed(x)
        x = module.seq(x)
        x = module.mul * x
        x = module.fc2(x)

        return x


class MyTModule(TModule):

    def forward(self, x):
        x = self.embed(x)
        x = self.seq(x)
        x = self.mul * x
        x = self.fc2(x)
        return x

    
    # def apply():

class JaxModule:

    @classmethod
    def init(cls, *args, **kwargs):

        module, t_module = cls.setup(*args, **kwargs)

        state = module.get_state()
        t_state = module.get_t_state()

        apply = functools.partial(cls.apply, module)

        return state, apply, t_module, t_state


class LayerNorm(JaxModule):
    def setup(normalized_shape, eps=1e-05):
        module = Module()
        module.config.eps = 1e-05

        t_ln = torch.nn.LayerNorm(normalized_shape, eps)

        t_weight = t_ln.weight
        j_weight = jnp.array(t_weight.detach().numpy())

        module.register_parameter("weight", j_weight, t_weight)

        t_bias = t_ln.bias
        j_bias = jnp.array(t_bias.detach().numpy())

        module.register_parameter("bias", j_bias, t_bias)

        return module, t_ln

    def apply(module, state, x):
        module = module(state)

        e_x = jnp.average(x, axis=-1, keepdims=True)
        v_x = jnp.average((x-e_x)**2, axis=-1, keepdims=True)

        ln = (x - e_x)/jnp.sqrt(v_x + module.config.eps) * module.weight + module.bias

        return ln




def test_ln():
    
    state, apply, t_ln, t_state = LayerNorm.init(2)

    x = np.array([[1.0, 2.0],
                  [2.0, 3.0]])

    t_x = torch.tensor(x, dtype=torch.float32)
    j_x = jnp.array(x)

    j_y = apply(state, j_x)
    t_y = t_ln(t_x)

    print(f"j_y: {j_y}")
    print(f"t_y: {t_y}")



class NextModule(JaxModule):

    # @classmethod
    # def init(cls, *args, **kwargs):

    #     module, t_module = cls.setup(*args, **kwargs)

    #     state = module.get_state()
    #     t_state = module.get_t_state()

    #     apply = functools.partial(cls.apply, module)

    #     return state, apply, t_module(module), t_state

    def setup(vocab, embed, dim_next, dim_out):

        module = Module()

        module.trunk = MyModule.init(vocab, embed, 10, 20, dim_next)

        t_bias = torch.normal(torch.zeros(dim_next))
        t_bias.requires_grad = True
        j_bias = t_bias.detach().numpy()
        
        module.register_parameter('next_bias', j_bias, t_bias)

        module.head = Linear.init(dim_next, dim_out)


        return module, NextTModule(module)

        # apply = functools.partial(NextModule.apply, module)

        # return state, apply, NextTModule(module), t_state

    def apply(module, state, x):
        module = module(state)


        x = module.trunk(x)
        x = jax.nn.relu(x)
        x = module.next_bias + x
        x = jax.nn.relu(x)
        x = module.head(x)

        return x


class NextTModule(TModule):

    def forward(self, x):
        x = self.trunk(x)
        x = torch.nn.functional.relu(x)
        # print(f"next bias: {self.next_bias},  x: {x}")
        x = self.next_bias + x
        x = torch.nn.functional.relu(x)
        x = self.head(x)

        return x


        

def test():

    apply = jax.jit(Linear.apply)
    apply_nojit = Linear.apply

    x = jnp.ones(10)

    s1, apply1, *_ = Linear.init(10, 20, True)
    s2, apply2, *_ = Linear.init(10, 20, True)
    s3, apply3, *_ = Linear.init(10, 20, False)

    apply1 = jax.jit(apply1)
    apply2 = jax.jit(apply2)
    apply3 = jax.jit(apply3)

    print(f"s1(x): {apply1(s1, x)}")

    print(f"s2(x): {apply2(s2, x)}")

    print(f"s3(x): {apply3(s3, x)}")


    print(f"s1(x): {apply_nojit(s1, x)}")

    print(f"s2(x): {apply_nojit(s2, x)}")

    print(f"s3(x): {apply_nojit(s3, x)}")
        
    e, e_apply, t_e, t_p = Embedding.init(5, 10)

    x = [[1,2],[3, 4]]
    t_x = torch.tensor(x)
    j_x = jnp.array(x)

    print(f"pytorch embed: {t_e(t_x)}")
    print(f"    jax embed: {e_apply(e, j_x)}")

    def sum_embed(state, x):
        return jnp.sum(e_apply(state, x))
    e_grad = jax.grad(sum_embed)

    print(f"grad: {e_grad(e, j_x)}")

    seq, seq_apply, t_seq, t_seq_p = Sequential.init(Embedding.init(5, 10), Linear.init(10, 20, True), Linear.init(20, 10, True))

    x = [1,2]
    t_x = torch.tensor(x)
    j_x = jnp.array(x)

    print(f"torch sequential: {t_seq(t_x)}")
    print(f"  jax sequential: {seq_apply(seq, j_x)}")


    # x = np.random.normal(np.zeros(2, dtype=np.float32))
    # t_x = torch.tensor(x,dtype=torch.float32)
    # print(t_x)
    # j_x = jnp.array(x)

    m_state, m_apply, t_module, t_m_p = NextModule.init(5, 10, 20, 2)


    # print(f"torch version: {t_module(t_x)}")
    # print(f"  jax version: {m_apply(m_state, j_x)}")

    # t_module.zero_grad()

    def loss(params, constants, x):
        return jnp.sum(m_apply({'params': params, 'constants': constants}, x))



    t_loss = torch.sum(t_module(t_x))

    t_loss.backward()

    t_grads = tree_map(lambda x: x.grad, t_m_p['params'])


    v = jax.value_and_grad(loss)

    j_l, j_grads = v(m_state['params'], m_state['constants'], j_x)

    diff = tree_map(lambda x,y : x.detach().numpy()-y, t_grads, j_grads)

    print(f"\n\npytorch loss: {t_loss}")
    print(f"\n\n    jax loss: {j_l}")

    print(f"\n\ndifference: {pprint.pformat(diff)}")



def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy between sets of logits and integer labels.
    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.
    References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
    Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Integers specifying the correct class for each input, with shape
        `[...]`.
    Returns:
    Cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
    """
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)

    no_ignore = jax.lax.stop_gradient(labels!=-100)

    ignore_labels = jnp.where(no_ignore, labels, jnp.zeros_like(labels))

    total = jax.lax.stop_gradient(jnp.sum(no_ignore))

    label_logits = jnp.take_along_axis(logits, ignore_labels[..., None], axis=-1)[..., 0]

    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    return jnp.sum(jnp.where(no_ignore, log_normalizers - label_logits, jnp.zeros_like(labels)))/total

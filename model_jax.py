'''
model.py
implements a simple self-attention layer with some custom stuff.
'''

from flax import jax_utils
from flax import linen
from flax import linen as nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.module import Module, compact, merge_param
from typing import Callable, Any, Optional
from einops import rearrange

t_Dense = lambda *args, **kwargs: nn.Dense(*args, kernel_init=jax.nn.initializers.he_uniform(), bias_init=jax.nn.initializers.he_uniform(in_axis=-1), **kwargs)

def Identity():
    # Is this function stupid? yes it is, but we keep it here so that 
    # it is easy to remove layers by replacing them with Identity.
    return lambda x: x

# equivalent of torch masked_fill
def masked_fill(to_fill, mask, fill):
    fill = jax.lax.broadcast(fill, to_fill.shape)
    # print(mask.shape)
    # print(to_fill.shape)
    mask = jax.lax.broadcast(mask, to_fill.shape)
    return jax.lax.select(mask, to_fill, fill)
# class ModelConfig:
#     def __init__(self, config):
#         self.config = config
        
        # vocab_size, context_length, n_layers, embedding_dim, n_heads=1, use_diag=True, **kwargs):
        # self.vocab_size = vocab_size
        # self.embedding_dim = embedding_dim
        # self.context_length = context_length
        # self.n_heads = n_heads
        # self.n_layers = n_layers
        # self.use_diag = use_diag
        # for k,v in kwargs.items():
        #     setattr(self, k, v)

class SelfAttention(nn.Module):
    config: Any

        

    @compact
    def __call__(self, x):
        config = self.config
        dim = config.embedding_dim

        key_matrix = t_Dense(config.embedding_dim) # maybe we want to mess with initialization schemes later?
        query_matrix = Identity()#torch.nn.Linear(self.dim, self.dim)
        value_matrix = t_Dense(config.embedding_dim)
        
        n_heads = config.n_heads
        context_length = config.context_length
        
        
        mask = self.variable('constants',
                             'causal_mask',
                             lambda : jnp.tril(jnp.ones((config.context_length, config.context_length))).reshape((1, 1, config.context_length, config.context_length)))
        

        assert config.embedding_dim % config.n_heads == 0, "number of heads ({}) does not evenly divide embedding dim ({})".format(config.n_heads, config.embedding_dim)

        *_, T, C = x.shape

        assert C == config.embedding_dim, "specified axis does not have correct dimension: was {}, expected {}".format(C, config.embedding_dim)

        split_heads_shape = x.shape[:-1] + (config.n_heads, config.embedding_dim // config.n_heads)
        key = rearrange(key_matrix(x).reshape(split_heads_shape), '... T nh hs -> ... nh T hs')#transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
        query = rearrange(query_matrix(x).reshape(split_heads_shape), '... T nh hs -> ... nh T hs') #.transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
        value = rearrange(value_matrix(x).reshape(split_heads_shape), '... T nh hs -> ... nh T hs') #.transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]

        assert key.shape[-2] == T, "shape mismatch: {}".format(key.shape)

        # print(key.shape)
        # print(query.shape)

        logits = jnp.matmul(key, rearrange(query, '... nh T hs -> ... nh hs T')) # [..., nh, T, hs] x [..., nh, hs, T] -> [..., nh, T, T]
        print(f"mask shape: {(mask.value[:, :, :T, :T] == 0).shape}")
        print(f"logits shape: {logits.shape}")
        broadcast_mask = jnp.broadcast_to((mask.value[:, :, :T, :T] == 0), logits.shape)
        print(f"broadcast shape: {broadcast_mask.shape}")

        masked_logits = jnp.where(broadcast_mask, -jnp.inf, logits)
        
        # masked_fill(logits, mask.value[:,:,:T,:T] == 0, float('-inf'))

        att_weights = nn.softmax(masked_logits, axis=-1)

        y = jnp.matmul(att_weights, value) #  [..., nh, T, T]  x [..., nh, T, hs] -> [..., nh, T, hs]
        y = rearrange(y, '... nh T hs -> ... T nh hs').reshape(x.shape) # [..., nh, T, hs] -> [..., T, nh, hs] -> [..., T, C]

        return y






class ResidualSelfAttention(nn.Module):
    config: Any


    @compact
    def __call__(self, x):
        config = self.config

        selfattention = SelfAttention(config)

        assert not config.use_diag, "condnorm not supported in JAX implementation"

        ln = nn.LayerNorm(config.embedding_dim)

        fc1 = t_Dense(2 * config.embedding_dim)
        fc2 = t_Dense(config.embedding_dim)


        y = ln(x)
        y = selfattention(y)
        if config.use_fc:
            y = fc1(y)
            y = nn.gelu(y)
            y = fc2(y)
        y = x + y
        return y



class StackedAttention(nn.Module):
    config: Any


    def setup(self):
        config = self.config

        self.features = nn.Sequential([ResidualSelfAttention(config) for _ in range(config.n_layers)])
        self.tok_embeddings = nn.Embed(
            config.vocab_size,
            config.embedding_dim,
            embedding_init=nn.initializers.variance_scaling(config.vocab_size, 'fan_in', 'normal', out_axis=0))
        self.pos_embeddings = self.param('position_embedding',
                                         lambda rng : jnp.zeros((1, config.context_length, config.embedding_dim)))

        self.head = nn.Dense(config.vocab_size, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)
        
    def get_targets(mask, idx, T):
        targets = idx[:,1:T+1]
        targets = jnp.where(mask, -100, targets)
        # print(targets)
        return targets

    def __call__(self, idx, mask, labels):
        """
        idx is 1-hot encoding integer tensor shape [B, T] entries are indices into vocab
        mask: currently unused
        targets is 1-hot encoding integer tensor shape [B, T], entries are indices into vocab for labels.
            ith entry of bth row of targets is label for ith prefix of idx in bth example in the batch.
        """

        # x is 1-hot encoding
        B, T = idx.shape

        T = min(T-1, self.config.context_length)

        tok_embd = self.tok_embeddings(idx[:, :T])

        pos_embd = self.pos_embeddings[:, :T, :]

        attention_input = nn.softmax(tok_embd + pos_embd, axis=-1) # input for attention layers: shape [B, T, C]

        features = self.features(attention_input)

        # if not compute_loss:
        #     return features

        logits = self.head(features) # shape [B, T, V]

        targets = labels[:, :T]#
        # targets = StackedAttention.get_targets(mask, idx, T)

        # cross entropy loss doesn't know about T, so we flatten the time dimension:
        # print("logits: ", logits.shape)
        # print("targets: ", targets.shape)
        
        logits_for_CE = logits.reshape(-1, logits.shape[-1]) # shape [BT, V]
        # print(f"logits for ce shape: {logits_for_CE.size()}, original logits shape: {logits.size()}, targets shape: {targets.size()}")
        targets_for_CE = targets.reshape(-1) # shape [BT]

        # logits_for_CE = logits_for_CE[targets_for_CE != -100]
        # targets_for_CE = targets_for_CE[targets_for_CE != -100]

        stopped_logits = jax.lax.stop_gradient(logits_for_CE)
        stopped_targets = jax.lax.stop_gradient(targets_for_CE)

        predictions = jnp.argmax(stopped_logits, 1)
        num_targets = jnp.sum(stopped_targets != -100)
        num_correct = jnp.sum(stopped_targets == predictions)
        accuracy = num_correct/num_targets

    #         predictions = torch.argmax(logits_for_CE, 1)
    #         num_targets = torch.sum(targets_for_CE != -100)
    #         num_correct = torch.sum(targets_for_CE == predictions)
    #         accuracy = num_correct/num_targets

        # loss = jnp.average(optax.softmax_cross_entropy_with_integer_labels(logits_for_CE, targets_for_CE))
        loss = softmax_cross_entropy_with_integer_labels(logits_for_CE, targets_for_CE)

        return features, loss, accuracy        
                    

def softmax_cross_entropy_with_integer_labels(logits, labels):
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


# from omegaconf import OmegaConf
# config = OmegaConf.load('config/meta_opt/offset_sgd.yaml').model
# config.vocab_size = 1000

# model = StackedAttention(config)

# model_state= model.init(jax.random.PRNGKey(0), jnp.ones([2,10],dtype=int), jnp.full([2,10],False), jnp.ones([2,10], dtype=int))

# print(model_state['constants'])

# features, loss, accuracy = model.apply(model_state, jnp.zeros([1,50],dtype=int), jnp.full([1,50],False), jnp.ones([1,50], dtype=int))

# # grad_fn = jax.grad(lambda *args: model.apply(*args)[1])

# # grad = grad_fn(model_state, jnp.zeros([1,50],dtype=int), jnp.full([1,50],False), jnp.ones([1,50], dtype=int))

# print(f"loss: {loss}, accuracy: {accuracy}")



# logits = np.array(
#     [[10.0 ,4.0 ,5],
#      [2, 2, 2],
#      [-1, -3, 4]]
# )

# labels = np.array(
#     [0, 2, -100]
# )

# s = softmax_cross_entropy_with_integer_labels(logits, labels)

# print(f"loss: {s}")

# import torch


# print(f" torch: ",torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)))

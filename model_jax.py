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


def Identity():
    # Is this function stupid? yes it is, but we keep it here so that 
    # it is easy to remove layers by replacing them with Identity.
    return lambda x: x

# equivalent of torch masked_fill
def masked_fill(to_fill, mask, fill):
    fill = jax.lax.broadcast(fill, to_fill.shape)
    print(mask.shape)
    print(to_fill.shape)
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

        key_matrix = nn.Dense(config.embedding_dim) # maybe we want to mess with initialization schemes later?
        query_matrix = Identity()#torch.nn.Linear(self.dim, self.dim)
        value_matrix = nn.Dense(config.embedding_dim)
        
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
        masked_logits = jnp.where(mask.value, -jnp.inf, logits)
        
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

        fc1 = nn.Dense(2 * config.embedding_dim)
        fc2 = nn.Dense(config.embedding_dim)


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

        config.vocab_size = 1000

        self.features = nn.Sequential([ResidualSelfAttention(config) for _ in range(config.n_layers)])
        self.tok_embeddings = nn.Embed(config.vocab_size, config.embedding_dim)
        self.pos_embeddings = self.variable('constants',
                                       'position_embedding',
                                       lambda : jnp.zeros((1, config.context_length, config.embedding_dim)))
        self.head = nn.Dense(config.vocab_size,
                        kernel_init=jax.nn.initializers.zeros,
                        bias_init=jax.nn.initializers.zeros)
        
    def get_targets(mask, idx, T):
        targets = idx[:,1:T+1]
        targets = jnp.where(mask, -100, targets)
        # print(targets)
        return targets

    def __call__(self, idx, labels, compute_loss=True):
        """
        idx is 1-hot encoding integer tensor shape [B, T] entries are indices into vocab
        targets is 1-hot encoding integer tensor shape [B, T], entries are indices into vocab for labels.
            ith entry of bth row of targets is label for ith prefix of idx in bth example in the batch.
        """

        # x is 1-hot encoding
        B, T = idx.shape

        T = min(T-1, self.config.context_length)

        tok_embd = self.tok_embeddings(idx[:, :T])

        pos_embd = self.pos_embeddings.value[:, :T, :]

        x = nn.softmax(tok_embd + pos_embd, axis=-1) # input for attention layers: shape [B, T, C]

        features = self.features(x)

        if not compute_loss:
            return features

        logits = self.head(features) # shape [B, T, V]

        targets = labels[:, :T]#
        # targets = StackedAttention.get_targets(mask, idx, T)

        # cross entropy loss doesn't know about T, so we flatten the time dimension:
        # print("logits: ", logits.shape)
        # print("targets: ", targets.shape)
        
        logits_for_CE = logits.reshape(-1, logits.shape[-1]) # shape [BT, V]
        # print(f"logits for ce shape: {logits_for_CE.size()}, original logits shape: {logits.size()}, targets shape: {targets.size()}")
        targets_for_CE = targets.reshape(-1) # shape [BT]

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

        loss = optax.softmax_cross_entropy_with_integer_labels(logits_for_CE, targets_for_CE)

        return features, loss, accuracy        
                    


    # def __init__(self, config):
    #     super().__init__()
    #     self.config = config
    #     self.features = torch.nn.Sequential(*[ResidualSelfAttention(config) for _ in range(config.n_layers)])
    #     self.tok_embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
    #     self.pos_embeddings = torch.nn.Parameter(torch.zeros(1, config.context_length, config.embedding_dim))
    #     self.head = torch.nn.Linear(config.embedding_dim, config.vocab_size)
    #     with torch.no_grad():
    #         self.head.weight.mul_(0.0)
    #         self.head.bias.mul_(0.0)


    # def get_targets(mask, idx, T):
    #     targets = idx[:,1:T+1]
    #     targets = targets.masked_fill(mask[:,1:T+1] == 0, -100)
    #     # print(targets)
    #     return targets


    # def forward(self, idx, mask, labels, compute_loss=True):
    #     """
    #     idx is 1-hot encoding integer tensor shape [B, T] entries are indices into vocab
    #     targets is 1-hot encoding integer tensor shape [B, T], entries are indices into vocab for labels.
    #         ith entry of bth row of targets is label for ith prefix of idx in bth example in the batch.
    #     """

    #     # x is 1-hot encoding
    #     B, T = idx.size()

    #     T = min(T-1, self.config.context_length)

    #     tok_embd = self.tok_embeddings(idx[:, :T])

    #     pos_embd = self.pos_embeddings[:, :T, :]

    #     x = F.softmax(tok_embd + pos_embd, dim=-1) # input for attention layers: shape [B, T, C]

    #     features = self.features(x)

    #     if not compute_loss:
    #         return features

    #     logits = self.head(features) # shape [B, T, V]

    #     targets = labels[:, :T]#
    #     # targets = StackedAttention.get_targets(mask, idx, T)

    #     # cross entropy loss doesn't know about T, so we flatten the time dimension:
    #     # print("logits: ", logits.shape)
    #     # print("targets: ", targets.shape)
        
    #     logits_for_CE = logits.reshape(-1, logits.size(-1)) # shape [BT, V]
    #     # print(f"logits for ce shape: {logits_for_CE.size()}, original logits shape: {logits.size()}, targets shape: {targets.size()}")
    #     targets_for_CE = targets.reshape(-1) # shape [BT]

    #     with torch.no_grad():
    #         predictions = torch.argmax(logits_for_CE, 1)
    #         num_targets = torch.sum(targets_for_CE != -100)
    #         num_correct = torch.sum(targets_for_CE == predictions)
    #         accuracy = num_correct/num_targets

    #     loss = F.cross_entropy(logits_for_CE, targets_for_CE)

    #     return features, loss, accuracy




# config = ModelConfig(vocab_size=4, context_length=3, num_layers=2, embedding_dim=4, n_heads=2)

# l = StackedAttention(config)

# idx = torch.tensor([ [2, 3, 0, 1],
#                      [0, 1, 2, 3],
#                      [3, 0, 1, 2],
#                      [1, 2, 3, 0] ])

# mask = 0*torch.tensor([[1, 1, 1, 1],
#                      [1, 1, 1, 1],
#                      [1, 1, 1, 1],
#                      [1, 1, 1, 1]])

# print(l(idx, mask)[1])

# x = torch.tensor([ [[1.0,  0.0,  0.0, 0.0],
#                     [0.0,  0.0,  0.0, 1.0],
#                     [0.0,  0.5,  0.5, 0.0]],

#                    [[0.5,  0.2,  0.1, 0.2],
#                     [0.0,  1.0,  0.0, 0.0],
#                     [0.0,  0.0,  1.0, 0.0]]])
# print(x.shape)

# y = l(x)
# print(y)


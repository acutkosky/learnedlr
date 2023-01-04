import jax
from jax import numpy as jnp
import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np

from einops import rearrange


import jax_from_pt as jpt


class SelfAttention(jpt.JaxModule):
    def setup(config):
        module = jpt.Module()
        module.config = config

        dim = config.embedding_dim

        module.key_matrix = jpt.Linear.init(dim, dim)
        module.query_matrix = jpt.Identity.init()
        module.value_matrix = jpt.Linear.init(dim, dim)

        t_mask = torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length)
        j_mask = jnp.array(t_mask.numpy())

        module.register_buffer("mask", j_mask, t_mask)

        return module, T_SelfAttention(module)

    def apply(module, state, x):
        module = module(state)

        config = module.config

        *_, T, C = x.shape

        split_heads_shape = x.shape[:-1] + (config.n_heads, config.embedding_dim // config.n_heads)

        key = rearrange(module.key_matrix(x).reshape(split_heads_shape), '... T nh hs -> ... nh T hs') #.transpose(-2, -3)
        query = rearrange(module.query_matrix(x).reshape(split_heads_shape), '... T nh hs -> ... nh hs T') 
        value = rearrange(module.value_matrix(x).reshape(split_heads_shape), '... T nh hs -> ... nh T hs') #.transpose(-2, -3)


        logits = jnp.matmul(key, query) # [..., nh, T, hs] x [..., nh, hs, T] -> [..., nh, T, T]

        broadcast_mask = jnp.broadcast_to((module.mask[:, :, :T, :T] == 0), logits.shape)
        masked_logits = jnp.where(broadcast_mask, -jnp.inf, logits)
        # logits.masked_fill(module.mask[:,:,:T,:T] == 0, float('-inf'))

        att_weights = jax.nn.softmax(masked_logits, axis=-1)

        y = jnp.matmul(att_weights, value) #  [..., nh, T, T]  x [..., nh, T, hs] -> [..., nh, T, hs]
        y = rearrange(y, '... nh T hs -> ... T nh hs').reshape(x.shape) # [..., nh, T, hs] -> [..., T, nh, hs] -> [..., T, C]

        return y       

class T_SelfAttention(jpt.TModule):

    def forward(self, x):

        # I don't totally understand why heads are a good idea, but apparently they are...
        *_, T, C = x.shape

        config = self.config

        # assert C == self.dim, "specified axis does not have correct dimension: was {}, expected {}".format(C, self.dim)

        split_heads_shape = x.shape[:-1] + (config.n_heads, config.embedding_dim // config.n_heads)
        key = self.key_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
        query = self.query_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
        value = self.value_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]

        assert key.shape[-2] == T, "shape mismatch: {}".format(key.shape)

        # print(key.shape)
        # print(query.shape)

        logits = torch.matmul(key, query.transpose(-1, -2)) # [..., nh, T, hs] x [..., nh, hs, T] -> [..., nh, T, T]
        masked_logits = logits.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att_weights = F.softmax(masked_logits, dim=-1)

        y = torch.matmul(att_weights, value) #  [..., nh, T, T]  x [..., nh, T, hs] -> [..., nh, T, hs]
        y = y.transpose(-2, -3).reshape(x.shape) # [..., nh, T, hs] -> [..., T, nh, hs] -> [..., T, C]

        return y






# class SelfAttention(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.dim = config.embedding_dim
#         self.key_matrix = torch.nn.Linear(self.dim, self.dim) # maybe we want to mess with initialization schemes later?
#         self.query_matrix = torch.nn.Identity()#torch.nn.Linear(self.dim, self.dim)

#         self.value_matrix = torch.nn.Linear(self.dim, self.dim)
        
#         self.n_heads = config.n_heads
#         self.context_length = config.context_length
#         self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

#         assert self.dim % self.n_heads == 0, "number of heads ({}) does not evenly divide embedding dim ({})".format(self.n_heads, self.dim)


#     def forward(self, x):

#         # I don't totally understand why heads are a good idea, but apparently they are...
#         *_, T, C = x.shape

#         assert C == self.dim, "specified axis does not have correct dimension: was {}, expected {}".format(C, self.dim)

#         split_heads_shape = x.shape[:-1] + (self.n_heads, self.dim // self.n_heads)
#         key = self.key_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
#         query = self.query_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
#         value = self.value_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]

#         assert key.shape[-2] == T, "shape mismatch: {}".format(key.shape)

#         # print(key.shape)
#         # print(query.shape)

#         logits = torch.matmul(key, query.transpose(-1, -2)) # [..., nh, T, hs] x [..., nh, hs, T] -> [..., nh, T, T]
#         masked_logits = logits.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

#         att_weights = F.softmax(masked_logits, dim=-1)

#         y = torch.matmul(att_weights, value) #  [..., nh, T, T]  x [..., nh, T, hs] -> [..., nh, T, hs]
#         y = y.transpose(-2, -3).reshape(x.shape) # [..., nh, T, hs] -> [..., T, nh, hs] -> [..., T, C]

#         return y


class ResidualSelfAttention(jpt.JaxModule):

    def setup(config):
        module = jpt.Module()
        module.config = config

        module.selfattention = SelfAttention.init(config)
        module.ln = jpt.LayerNorm.init(config.embedding_dim)
        
        module.fc1 = jpt.Linear.init(config.embedding_dim, 2*config.embedding_dim)
        module.fc2 = jpt.Linear.init(2*config.embedding_dim, config.embedding_dim)

        return module, T_ResidualSelfAttention(module)

    def apply(module, state, x):
        module = module(state)

        y = module.ln(x)
        y = module.selfattention(y)

        if module.config.use_fc:
            y = module.fc1(y)
            y = jax.nn.gelu(y)
            y = module.fc2(y)

        y = x + y
        return y

class T_ResidualSelfAttention(jpt.TModule):
    def forward(self, x):
        y = self.ln(x)
        y = self.selfattention(y)
        
        if self.config.use_fc:
            y = self.fc1(y)
            y = F.gelu(y)
            y = self.fc2(y)

        y = x + y
        return y




# from omegaconf import OmegaConf
# config = OmegaConf.load('config/jax_test/test.yaml').model
# config.vocab_size = 1000

# state, apply, t_module, t_state = StackedAttention.init(config)

# x = np.random.normal(np.ones((10, config.context_length, config.embedding_dim)))

# idx = np.ones((10, config.context_length), dtype=int)
# mask = idx
# labels = np.zeros((10, config.context_length), dtype=int)

# t_idx = torch.tensor(idx, dtype=int)
# t_labels = torch.tensor(labels)
# t_mask = torch.tensor(mask)

# j_idx = 

# t_x = torch.tensor(x, dtype=torch.float32)
# j_x = jnp.array(x)

# t_y = t_module(t_x)
# j_y = apply(state, j_x)

# print(f"torch: {t_y}")
# print(f"jax: {j_y}")

# print(jnp.sum((t_y.detach().numpy() - j_y)**2))




class StackedAttention(jpt.JaxModule):
    def setup(config):
        module = jpt.Module()
        module.config = config

        module.features = jpt.Sequential.init(*[ResidualSelfAttention.init(config) for _ in range(config.n_layers)])
        module.tok_embeddings = jpt.Embedding.init(config.vocab_size, config.embedding_dim)
        t_pos_embeddings = torch.zeros(1, config.context_length, config.embedding_dim, requires_grad=True)
        j_pos_embeddings = jnp.array(t_pos_embeddings.detach().numpy())
        module.register_parameter("pos_embeddings", j_pos_embeddings, t_pos_embeddings)

        module.head = jpt.Linear.init(config.embedding_dim, config.vocab_size)

        module.head[0]['params']['weight'] = jnp.zeros_like(module.head[0]['params']['weight'])
        module.head[0]['params']['bias'] = jnp.zeros_like(module.head[0]['params']['bias'])

        with torch.no_grad():
            module.head[2].weight.mul_(0.0)
            module.head[2].bias.mul_(0.0)

    
        return module, T_StackedAttention(module)
            


    def apply(module, state, idx, mask, labels, compute_loss=True):
        module = module(state)

        B, T = idx.shape

        T = min(T-1, module.config.context_length)

        tok_embd = module.tok_embeddings(idx[:, :T])
        pos_embd = module.pos_embeddings[:, :T, :]
 
        x = jax.nn.softmax(tok_embd + pos_embd, axis=-1) # input for attention layers: shape [B, T, C]

        features = module.features(x)

        if not compute_loss:
            return features

        logits = module.head(features) # shape [B, T, V]

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
        loss = jpt.softmax_cross_entropy(logits_for_CE, targets_for_CE)

        return features, loss, accuracy  


class T_StackedAttention(jpt.TModule):
    def forward(self, idx, mask, labels, compute_loss=True):

        # x is 1-hot encoding
        B, T = idx.size()

        T = min(T-1, self.config.context_length)

        tok_embd = self.tok_embeddings(idx[:, :T])

        pos_embd = self.pos_embeddings[:, :T, :]

        x = F.softmax(tok_embd + pos_embd, dim=-1) # input for attention layers: shape [B, T, C]

        features = self.features(x)

        if not compute_loss:
            return features

        logits = self.head(features) # shape [B, T, V]

        targets = labels[:, :T]#
        # targets = StackedAttention.get_targets(mask, idx, T)

        # cross entropy loss doesn't know about T, so we flatten the time dimension:
        # print("logits: ", logits.shape)
        # print("targets: ", targets.shape)
        
        logits_for_CE = logits.reshape(-1, logits.size(-1)) # shape [BT, V]
        # print(f"logits for ce shape: {logits_for_CE.size()}, original logits shape: {logits.size()}, targets shape: {targets.size()}")
        targets_for_CE = targets.reshape(-1) # shape [BT]

        with torch.no_grad():
            predictions = torch.argmax(logits_for_CE, 1)
            num_targets = torch.sum(targets_for_CE != -100)
            num_correct = torch.sum(targets_for_CE == predictions)
            accuracy = num_correct/num_targets

        loss = F.cross_entropy(logits_for_CE, targets_for_CE)

        return features, loss, accuracy



# from omegaconf import OmegaConf
# config = OmegaConf.load('config/jax_test/test.yaml').model
# config.vocab_size = 1000

# state, apply, t_module, t_state = StackedAttention.init(config)

# x = np.random.normal(np.ones((10, config.context_length, config.embedding_dim)))

# idx = np.ones((10, config.context_length), dtype=int)
# mask = idx
# labels = np.zeros((10, config.context_length), dtype=int)

# t_idx = torch.tensor(idx, dtype=int)
# t_labels = torch.tensor(labels)
# t_mask = torch.tensor(mask)

# j_idx = jnp.array(idx)
# j_mask = jnp.array(mask)
# j_labels = jnp.array(labels)

# lr = 0.001



# t_adamw = torch.optim.AdamW(t_module.parameters(), lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.001)

# t_f, t_l, t_a = t_module(t_idx, t_mask, t_labels)
# j_f, j_l, j_a = apply(state, j_idx, j_mask, j_labels)

# t_l.backward()
# t_adamw.step()



# import optax

# j_adamw = optax.adamw(lr, b1=0.9, b2=0.99, eps=1e-08, eps_root=0.0, weight_decay=0.001)
# j_adamw_state = j_adamw.init(state['params'])

# def loss_fn(params, constants, idx, mask, labels):
#     return apply({'params': params, 'constants': constants}, idx, mask, labels)[1]

# grad_fn = jax.grad(loss_fn)

# params = state['params']
# constants = state['constants']

# grads = grad_fn(params, constants, j_idx, j_mask, j_labels)

# updates, j_adamw_state = j_adamw.update(grads, j_adamw_state, params)
# params = optax.apply_updates(params, updates)

# state = {
#     'params': params,
#     'constants': constants
# }


# n_t_f, n_t_l, n_t_a = t_module(t_idx, t_mask, t_labels)
# n_j_f, n_j_l, n_j_a = apply(state, j_idx, j_mask, j_labels)
    



# print(f"torch: {t_l}")
# print(f"jax: {j_l}")

# print(f"new torch: {n_t_l}")
# print(f"new jax: {n_j_l}")

# t_f = t_f.detach().numpy()

# n_t_f = n_t_f.detach().numpy()

# print(jnp.sum((t_f - j_f)**2)/ jnp.sum(t_f**2 + j_f**2))

# print(jnp.sum((n_t_f - n_j_f)**2)/ jnp.sum(n_t_f**2 + n_j_f**2))






# class StackedAttention(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.features = torch.nn.Sequential(*[ResidualSelfAttention(config) for _ in range(config.n_layers)])
#         self.tok_embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
#         self.pos_embeddings = torch.nn.Parameter(torch.zeros(1, config.context_length, config.embedding_dim))
#         self.head = torch.nn.Linear(config.embedding_dim, config.vocab_size)
#         with torch.no_grad():
#             self.head.weight.mul_(0.0)
#             self.head.bias.mul_(0.0)


#     def get_targets(mask, idx, T):
#         targets = idx[:,1:T+1]
#         targets = targets.masked_fill(mask[:,1:T+1] == 0, -100)
#         # print(targets)
#         return targets


#     def forward(self, idx, mask, labels, compute_loss=True):
#         """
#         idx is 1-hot encoding integer tensor shape [B, T] entries are indices into vocab
#         targets is 1-hot encoding integer tensor shape [B, T], entries are indices into vocab for labels.
#             ith entry of bth row of targets is label for ith prefix of idx in bth example in the batch.
#         """

#         # x is 1-hot encoding
#         B, T = idx.size()

#         T = min(T-1, self.config.context_length)

#         tok_embd = self.tok_embeddings(idx[:, :T])

#         pos_embd = self.pos_embeddings[:, :T, :]

#         x = F.softmax(tok_embd + pos_embd, dim=-1) # input for attention layers: shape [B, T, C]

#         features = self.features(x)

#         if not compute_loss:
#             return features

#         logits = self.head(features) # shape [B, T, V]

#         targets = labels[:, :T]#
#         # targets = StackedAttention.get_targets(mask, idx, T)

#         # cross entropy loss doesn't know about T, so we flatten the time dimension:
#         # print("logits: ", logits.shape)
#         # print("targets: ", targets.shape)
        
#         logits_for_CE = logits.reshape(-1, logits.size(-1)) # shape [BT, V]
#         # print(f"logits for ce shape: {logits_for_CE.size()}, original logits shape: {logits.size()}, targets shape: {targets.size()}")
#         targets_for_CE = targets.reshape(-1) # shape [BT]

#         with torch.no_grad():
#             predictions = torch.argmax(logits_for_CE, 1)
#             num_targets = torch.sum(targets_for_CE != -100)
#             num_correct = torch.sum(targets_for_CE == predictions)
#             accuracy = num_correct/num_targets

#         loss = F.cross_entropy(logits_for_CE, targets_for_CE)

#         return features, loss, accuracy




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


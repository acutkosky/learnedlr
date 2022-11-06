'''
model.py
implements a simple self-attention layer with some custom stuff.
'''

import torch
from torch.nn import functional as F
import condnorm

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

class SelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.embedding_dim
        self.key_matrix = torch.nn.Linear(self.dim, self.dim) # maybe we want to mess with initialization schemes later?
        self.query_matrix = torch.nn.Identity()#torch.nn.Linear(self.dim, self.dim)

        self.value_matrix = torch.nn.Linear(self.dim, self.dim)
        
        self.n_heads = config.n_heads
        self.context_length = config.context_length
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

        assert self.dim % self.n_heads == 0, "number of heads ({}) does not evenly divide embedding dim ({})".format(self.n_heads, self.dim)


    def forward(self, x):

        # I don't totally understand why heads are a good idea, but apparently they are...
        *_, T, C = x.shape

        assert C == self.dim, "specified axis does not have correct dimension: was {}, expected {}".format(C, self.dim)

        split_heads_shape = x.shape[:-1] + (self.n_heads, self.dim // self.n_heads)
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





class ResidualSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.selfattention = SelfAttention(config)
        self.diagnorm = condnorm.DiagNorm(momentum=0.0001, ignore_dims=[1])
        self.ln = torch.nn.LayerNorm(config.embedding_dim)
        self.scaling = torch.nn.Parameter(torch.zeros(1))
        # self.register_parameter("residual_weight", 
        # self.residual_weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if self.config.use_diag == 'true':
            y = self.diagnorm(x)
        y = self.ln(x)
        y = self.selfattention(y)
        y = x + y#*self.scaling#*(1-self.residual_weight) + self.residual_weight*y
        return y

class StackedAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.features = torch.nn.Sequential(*[ResidualSelfAttention(config) for _ in range(config.n_layers)])
        self.tok_embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embeddings = torch.nn.Parameter(torch.zeros(1, config.context_length, config.embedding_dim))
        self.head = torch.nn.Linear(config.embedding_dim, config.vocab_size)
        with torch.no_grad():
            self.head.weight.mul_(0.0)
            self.head.bias.mul_(0.0)


    def get_targets(mask, idx, T):
        targets = idx[:,1:T+1]
        targets = targets.masked_fill(mask[:,1:T+1] == 0, -100)
        # print(targets)
        return targets


    def forward(self, idx, mask, labels, compute_loss=True):
        """
        idx is 1-hot encoding integer tensor shape [B, T] entries are indices into vocab
        targets is 1-hot encoding integer tensor shape [B, T], entries are indices into vocab for labels.
            ith entry of bth row of targets is label for ith prefix of idx in bth example in the batch.
        """

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


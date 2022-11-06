
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import functools
import random

IDCOUNT=0

SMALL_VALUE=1e-5

class DiagNorm(nn.Module):
    def __init__(self, momentum=0.1, eps=SMALL_VALUE, ignore_dims=[]):
        super(DiagNorm, self).__init__()
        global IDCOUNT
        self.eps = eps
        self.scale =0
        self.mean = 0
        self.total = 0
        self.momentum=momentum
        self.ignore_dims = [0]+ignore_dims
        self.id = IDCOUNT
        IDCOUNT += 1



    def forward(self, x):
        dims = x.dim()
        B, *D = x.shape
        # increment = functools.reduce(lambda x,y: x*y, D)
        self.total += B


        # with torch.no_grad():
        # mean = torch.sum(x, dim=[0,2,3], keepdim=True)/(B*H*W)
        # var =  torch.sum((x-mean)**2, dim=[0,2,3], keepdim=True)/(B*H*W)
        if self.training:
            with torch.no_grad():
                mean = torch.mean(x, dim=self.ignore_dims, keepdim=True)
                variance =  torch.mean(x**2, dim=self.ignore_dims, keepdim=True)

                momentum = max(B/self.total, self.momentum)
                # if self.scale is None:
                #     self.scale = variance
                #     self.mean = mean
                # else:
                # try:
                #     print(f"id: {self.id}, x shape: {x.shape}, momentum: {momentum}, scale shape: {self.scale.shape}, variance shape: {variance.shape}")
                # except AttributeError:
                #     print(f"id; {self.id}, x hape: {x.shape}, momentum: {momentum}, scale: {self.scale}, variance shape: {variance.shape}")
                self.scale = self.scale * (1-momentum) + momentum*variance
                self.mean = self.mean * (1-momentum) + momentum*mean

        return (x-self.mean) *torch.rsqrt(self.eps + self.scale - self.mean**2 + B/self.total)


class DimensionalNorm(nn.Module):
    def __init__(self, eps=SMALL_VALUE):
        super(DimensionalNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        dims = x.dim()

        x_sq = x**2

        x_div = torch.ones_like(x)

        for d in range(1, dims):
            x_div *= torch.rsqrt(self.eps + torch.sum(x_sq, dim=d, keepdim=True))

        x_div = x_div ** (1.0/(dims-1))

        return x * x_div





class CondNorm2d(nn.Module):

    def __init__(self, channels, use_bias=True, affine=True, eps=SMALL_VALUE, momentum=0.1):
        super(CondNorm2d, self).__init__()

        self.channels = channels
        self.eps = eps
        self.use_bias = use_bias
        self.momentum = momentum
        self.register_buffer('mean', torch.zeros((1, channels, 1, 1)))
        self.register_buffer('variance', torch.zeros((1, channels, 1, 1)))
        self.register_buffer('count', torch.zeros(1))
        self.affine = affine
        if affine:
            self.scale = nn.Parameter(torch.ones((1, channels, 1, 1)))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))


    def train(self, mode):
        super(CondNorm2d, self).train(mode)


    def forward(self, X):

        # we're going to do a kind of 'improper' update style thing even if 
        # in eval mode:
        B, C, H, W = X.shape
        per_example_count_increment = B* H * W #+ torch.sqrt(self.count)
        # with torch.no_grad():
        per_example_new_count = self.count + per_example_count_increment

        momentum = torch.clamp(per_example_count_increment/per_example_new_count, min=self.momentum)
        # X_mean = torch.einsum('b c h w -> c', X)
        X_mean = torch.mean(X, dim=[0,2,3], keepdim=True)
        # X_mean = torch.einsum('b c h w -> c', X) / count_increment
        

        per_example_mean = self.mean * (1- momentum) + momentum * X_mean 


        X_variance = torch.mean(X**2, dim=[0,2,3], keepdim=True)
        # new_mean = self.mean + (X_mean - self.mean) * count_increment/new_count

        # expanded_mean = einops.rearrange(new_mean, '(b c h w) -> b c h w', b=1, h=1, w=1)
        # X_variance = torch.einsum('b c h w -> c', (X - expanded_mean)**2)


        # expanded_X_mean = einops.rearrange(X_mean, '(b c h w) -> b c h w', b=1, h=1, w=1)
        # X_variance = torch.einsum('b c h w -> c', (X - expanded_X_mean)**2) / count_increment
        # X_variance = torch.mean((X-expanded_X_mean)**2, dim=[0,2,3])

        per_example_variance = self.variance * (1- momentum) + momentum * X_variance

        # new_variance = self.variance + (X_variance - self.variance)*count_increment/new_count

        # expanded_variance = einops.rearrange(new_variance, '(b c h w) -> b c h w', b=1, h=1, w=1)

        # expanded_X_variance = einops.rearrange(X_variance, '(b c h w) -> b c h w', b=1, h=1, w=1)

        X = (X - per_example_mean)*torch.rsqrt(per_example_variance - per_example_mean**2 + self.eps)
        if self.affine:
            X = self.scale * X
            if self.use_bias:
                X += self.bias



        # update statistics if in training mode
        if self.training:
            with torch.no_grad():
                count_increment = B * H * W
                new_count = self.count + count_increment
                momentum = torch.clamp(count_increment/new_count, min=self.momentum)
                full_X_mean = torch.mean(X, dim=[0, 2,3], keepdim=True)
                full_X_variance = torch.mean(X**2, dim=[0, 2,3], keepdim=True)
                new_mean = self.mean * (1- momentum) + momentum * full_X_mean 
                new_variance = self.variance * (1- momentum) + momentum * full_X_variance
                self.count.copy_(new_count)
                self.mean.copy_(new_mean)
                self.variance.copy_(new_variance)

        

        return X
        



if __name__ == '__main__':
    picture = torch.tensor([ [[[0.0,0.0], [0.0,0.0]],
                              [[1.0,1.0], [1.0,1.0]]],
                             [[[1.0,1.0], [1.0,1.0]],
                              [[1.0,1.0], [1.0,1.0]]] ])
    picture2 = torch.tensor([ [[[5.0,0.0], [0.0,0.0]],
                               [[1.0,1.0], [1.0,1.0]]],
                              [[[1.0,1.0], [1.0,1.0]],
                               [[1.0,1.0], [1.0,1.0]]] ])

    B = 10
    C = 10
    H = 10
    N = 10
    picture_big= torch.randn((B, C, H, N))
    picture2_big = torch.randn((B, C, H, N))
    print(picture.shape)

    bn = torch.nn.BatchNorm2d(C,eps=SMALL_VALUE, affine=False)
    cn = CondNorm2d(C,eps=SMALL_VALUE, affine=False, momentum=1.0)
    dn = DiagNorm(momentum=1.0, eps=SMALL_VALUE)

    bn.train(True)
    cn.train(True)
    # print(bn(picture) - cn(picture))
    # print(bn(picture2) - cn(picture2))

    # print(bn.weight)
    # print(cn.scale)

    # sbn = torch.optim.SGD(bn.parameters(), lr=0.1)
    # scn = torch.optim.SGD(cn.parameters(), lr=0.1)

    # pbn = torch.sum(bn(picture_big))
    # pbn = torch.sum(bn(picture_big))
    # pbn = torch.sum(bn(picture_big))
    # pbn = torch.sum(bn(picture_big))
    # pbn = torch.sum(bn(picture_big))
    # pbn.backward()
    # sbn.step()

    # pcn = torch.sum(cn(picture_big))
    # pcn = torch.sum(cn(picture_big))
    # pcn = torch.sum(cn(picture_big))
    # pcn = torch.sum(cn(picture_big))
    # pcn = torch.sum(cn(picture_big))
    # pcn.backward()
    # scn.step()

    # print(bn(picture) - cn(picture))
    print(cn(picture2_big) - bn(picture2_big))



    # print(cn(picture))
    # print(cn(picture2))
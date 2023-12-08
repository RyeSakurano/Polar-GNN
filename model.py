from locale import normalize
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from dgl import function as fn
import dgl.ops as ops
from dgl.nn.pytorch.conv import GraphConv, GATConv
import numpy as np
import math

class AngleLoss(nn.Module):
    def __init__(self, m=0.5):
        super(AngleLoss, self).__init__()
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

    def forward(self, input, target):
        cos_theta = input.gather(dim=1, index=target.view(-1, 1))
        sin_theta = torch.sqrt(1 - cos_theta ** 2)
        new_cos = cos_theta * self.cos_m - sin_theta * self.sin_m
        input = torch.scatter(input, dim=1, index=target.view(-1, 1), src=new_cos)
        return F.cross_entropy(input, target)


# Convolution Block
class PolarConvLayer(nn.Module):
    def __init__(self):
        super(PolarConvLayer, self).__init__()

    def forward(self, g, mode):
        if mode == 'polar':
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            feat = g.ndata['h']

        else:
            # gcn
            deg = g.out_degrees().float()
            norm = torch.pow(deg, -0.5).unsqueeze(dim=1)
            feat = g.ndata['h']
            feat = feat * norm
            g.ndata['h'] = feat
            g.update_all(fn.copy_src('h', 'm'), fn.sum(msg='m', out='h'))
            feat = g.ndata['h']
            feat = norm * feat
            g.ndata['h'] = feat 
        
        return feat


class PolarGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, str_dim, rep_dim, out_dim, fea_drop=0.5, weight_drop=0.5, num_layer=2, 
                    lamb = 0.5, eta = 1.0, num_head=3, mode='polar', struct='polar'):
        super(PolarGCN, self).__init__()
        
        self.num_layer = num_layer
        self.fea_drop = fea_drop
        self.weight_drop = weight_drop
        self.num_head = num_head
        self.eps = 1e-8
        self.labmda = lamb
        self.eta = eta
        self.mode = mode
        self.fisrt_epoch = True 
        self.prelu = nn.LeakyReLU(negative_slope=0.05)

        self.layers = nn.ModuleList([PolarConvLayer()] * self.num_layer)
        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Parameter(torch.FloatTensor(size=(hidden_dim, out_dim)))
        self.w1 = nn.ParameterList([nn.Parameter(torch.FloatTensor(hidden_dim))] * self.num_head)
        self.w2 = nn.ParameterList([nn.Parameter(torch.FloatTensor(str_dim))] * self.num_head)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2, gain=1.414)
        for i in range(self.num_head):
            nn.init.ones_(self.w1[i])
            nn.init.ones_(self.w2[i])
    
    def edge_applying(self, edges):
        # input features similarity
        weight_sum1 = 0
        for i in range(self.num_head):
            h_src = self.w1[i] * F.dropout(edges.src['h'],p=self.weight_drop, training=self.training)
            h_dst = self.w1[i] * F.dropout(edges.dst['h'],p=self.weight_drop, training=self.training)
            h_src = F.normalize(h_src, dim=1)
            h_dst = F.normalize(h_dst, dim=1)
            weight = torch.einsum('ij, ij->i', h_src, h_dst)
            weight_sum1 += weight
        weight1 = weight_sum1 / self.num_head

        # input structures similarity
        weight_sum2 = 0
        for i in range(self.num_head):
            h_src = self.w2[i] * F.dropout(edges.src['pos'], p=self.weight_drop, training=self.training)
            h_dst = self.w2[i] * F.dropout(edges.dst['pos'], p=self.weight_drop, training=self.training)
            h_src = F.normalize(h_src, dim=1)
            h_dst = F.normalize(h_dst, dim=1)
            weight = torch.einsum('ij, ij->i', h_src, h_dst)
            weight_sum2 += weight
        weight2 = weight_sum2 / self.num_head

        weight = self.labmda * weight1 + (1 - self.labmda) * weight2
        return {'w': weight}

    def forward(self, g, h):
        h = self.t1(h)
        h = F.dropout(h, p=self.fea_drop, training=self.training)
        h = self.prelu(h)
        if self.mode == 'polar':
            h = F.normalize(h, p=2, dim=1, eps=self.eps)
            g.ndata['h'] = h
            g.apply_edges(self.edge_applying)
        else:
            g.ndata['h'] = h

        for i in range(self.num_layer):
            h = self.layers[i](g, self.mode)
            if self.mode == 'polar':
                h = F.normalize(h, p=2, dim=1, eps=self.eps)
            g.ndata['h'] = h
        g.ndata['rep'] = h
        self.fisrt_epoch = False

        h = F.dropout(h, p=self.fea_drop, training=self.training)
        h = torch.relu(h)
        if self.mode == 'polar':
            h = F.normalize(h, p=2, dim=1)
            output = torch.matmul(h, F.normalize(self.t2, dim=0, eps=self.eps))
        else:
            output = torch.mm(h, self.t2)
        logits = F.softmax(output, dim=1)
        return output, logits, h

import argparse, time, math
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
import dgl.function as fn
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

class GCNLayer_ST(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation,
                 new_dilation,
                 skip_channels,
                 kernel_size=2,
                 bias=True,
                 adaptadj=True):
        super(GCNLayer_ST, self).__init__()
        self.adaptadj=True
        self.filter_convs = nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=(1, kernel_size), dilation=new_dilation)

        self.gate_convs = nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(1, kernel_size), dilation=new_dilation)

        # 1x1 convolution for residual connection
        self.residual_convs= nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(1, 1))

        # 1x1 convolution for skip connection
        self.skip_convs = (nn.Conv1d(in_channels=out_channels,
                                         out_channels=skip_channels,
                                         kernel_size=(1, 1)))

        self.bn = (nn.BatchNorm2d(out_channels))

        self.gconv = (GraphConv(out_channels, out_channels, activation=activation))

    def forward(self, g, h, skip):
        residual = h
        # dilated convolution
        filter = self.filter_convs(residual)
        filter = th.tanh(filter)
        gate = self.gate_convs(residual)
        gate = th.sigmoid(gate)
        h = filter * gate

        s = h
        s = self.skip_convs(s)
        if skip is not None:
            skip = skip[:, :, :, -s.size(3):]
        else:
            skip = 0
        skip = s + skip

        h = h.permute(0, 2, 3, 1)
        batch_size = h.size(0)
        h = h.reshape(-1, h.size(2), h.size(3))
        if self.adaptadj: 
            h = self.gconv(g, h, edge_weight=g.edata['edge_weights'])
        else:
            h = self.gconv(g, h)
            
        h = h.reshape(batch_size, -1, h.size(1), h.size(2))
        h = h.permute(0, 3, 1, 2)
        h = h + residual[:, :, :, -h.size(3):]
        h = self.bn(h)

        return h, skip


class GCN_ST(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 _,
                 n_hidden,
                 dropout,
                 skip_option,
                 num_nodes, 
                 in_channels=1,
                 kernel_size=2,
                 adaptadj=True,
                 adj_embed=10
                 ):
        super(GCN_ST, self).__init__()
        skip_channels = 256
        end_channels = 512
        layers = 2
        self.adaptadj = adaptadj
        if in_feats == 12:
            blocks = 4
        elif in_feats == 4:
            blocks = 1
        self.dropout = dropout
        self.skip_option = skip_option   
        # input layer
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=n_hidden, kernel_size=(1,1))

        # hidden layers
        self.layers = nn.ModuleList()
        receptive_field = 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.layers.append(GCNLayer_ST(n_hidden, n_hidden, nn.ReLU(), new_dilation, skip_channels,kernel_size=2,
                 bias=True, adaptadj=adaptadj))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        # ****output layer*****
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_1_no_skip = nn.Conv2d(in_channels=n_hidden,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_feats,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.nodevec1 = nn.Parameter(
            th.randn(num_nodes, adj_embed), requires_grad=True
        )
        self.nodevec2 = nn.Parameter(
            th.randn(adj_embed, num_nodes), requires_grad=True
        )

    def forward(self, g, features):
        features = features.transpose(1, 3)
        if self.adaptadj:
            g = self._update_edge(g)
        in_len = features.size(-1)
        if in_len < self.receptive_field:
            h = nn.functional.pad(features, (self.receptive_field - in_len, 0, 0, 0))
        else:
            h = features
        h = self.start_conv(h)

        skip = None
        for i, layer in enumerate(self.layers):
            h, skip = layer(g, h, skip)

        # Skip option (original). Comment for Sequential option.
        if self.skip_option:
            h = F.relu(skip)
            h = F.relu(self.end_conv_1(h))
        else:
            h = F.relu(self.end_conv_1_no_skip(h))
        h = self.end_conv_2(h)
        return h.squeeze(-1)

    def _update_edge(self, g):
        graphs = dgl.unbatch(g)
        for graph in graphs:
            adj = graph.adj().to_dense().bool().to(g.device.type)
            adj_mask = (~adj) * -1e9
            adaptadj = F.relu(self.nodevec1.mm(self.nodevec2)) + adj_mask
            adaptadj = F.softmax(adaptadj, dim=1)
            newadj = adj + adaptadj
            eweight = newadj[newadj > 0]
            graph.edata['edge_weights'] = eweight
        return dgl.batch(graphs)
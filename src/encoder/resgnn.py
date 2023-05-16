import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder.gnn import GNN_layer
import dgl
from dgl.nn.pytorch import Sequential


class ResGNN_layer(nn.Module): 
    def __init__(
        self, 
        in_feat: int, 
        out_feat: int, 
        identity: bool = True,
        adaptadj: bool = False
    ) -> None:
        super(ResGNN_layer, self).__init__()
        
        self.gnn_layer = GNN_layer(in_feat, out_feat, adaptadj)
        self.identity = identity
        if not self.identity:
            self.linear = nn.Linear(in_feat, 2 * out_feat, bias=False)

    def forward(self, g, h): 
        out = self.gnn_layer(g, h)
        if self.identity:
            return out + h
        else:
            return out + self.linear(h)

          
class ResGNN(nn.Module): 
    def __init__(
        self,
        in_feat: int, 
        out_feat: int, 
        n_layers: int, 
        hidden_size: int, 
        num_nodes: int, 
        identity_init: bool = False,
        identity_mid: bool = True,
        adaptadj: bool = False,
        adaptembed: int = 10,  
    ) -> None:
        super(ResGNN, self).__init__()

        modules = [] 
        in_size = in_feat
        identity = identity_init
        for _ in range(n_layers): 
            modules.append(ResGNN_layer(in_size, hidden_size, identity, adaptadj))
            in_size = 2 * hidden_size
            identity = identity_mid
        self.model = Sequential(*modules)
        self.linear_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, in_size), 
            nn.ReLU(),
            nn.Linear(in_size, out_feat)
        )

        self.adaptadj = adaptadj 
        self.nodevec1 = nn.Parameter(
            torch.randn(num_nodes, adaptembed), requires_grad=True
        )
        self.nodevec2 = nn.Parameter(
            torch.randn(adaptembed, num_nodes), requires_grad=True
        )

    def forward(self, g, h):
        if self.adaptadj: 
            g = self._update_edge(g)

        h = h[..., 0]
        shape = h.shape
        h = h.transpose(1,2).flatten(0,1)
        h = self.model(g, h)
        h = self.linear_out(h)
        return h.reshape(shape[0], shape[2], shape[1]).transpose(1,2)

    def _update_edge(self, g):
        graphs = dgl.unbatch(g)

        graph = graphs[0]
        adj = graph.adj().to_dense().bool().to(g.device.type)
        adj_mask = (~adj) * -1e9
        adaptadj = F.relu(self.nodevec1.mm(self.nodevec2)) + adj_mask
        adaptadj = F.softmax(adaptadj, dim=1)
        # Sparse sum
        eweight = adaptadj[adj > 0] + adj[adj > 0]
        for graph in graphs:
            graph.edata['edge_weights'] = eweight
        return dgl.batch(graphs)
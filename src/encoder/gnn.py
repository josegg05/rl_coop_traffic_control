from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import Sequential
import src.infrastructure.pytorch_utils as ptu


class MLP(nn.Module): 
    def __init__(self, in_feat, out_feat) -> None:
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_feat, out_feat), 
            nn.LayerNorm(out_feat), 
            nn.ReLU(),
        )

    def forward(self, x): 
        return self.model(x)


class GNN_layer(nn.Module): 
    def __init__(self, in_feat, out_feat, adaptadj=False) -> None:
        super(GNN_layer, self).__init__()
        self.mlp = MLP(in_feat, out_feat)
        self.adaptadj = adaptadj

    def forward(self, g, h):
        h = self.mlp(h)

        # Message passing and aggregation
        with g.local_scope(): 
            g.ndata['h'] = h
            
            aggregate_fn = fn.copy_u('h', 'm')
            if self.adaptadj:
                aggregate_fn = fn.u_mul_e('h', 'edge_weights', 'm')

            # update_all is a message passing API.
            g.update_all(
                message_func = aggregate_fn,
                reduce_func = fn.max('m', 'h_N')
            )
            h_N = g.ndata['h_N']
            return torch.concat([h, h_N], 1)


class GNN(nn.Module): 
    def __init__(
        self, 
        in_feat: int, 
        out_feat: int, 
        n_layers: int, 
        hidden_size: int,
        adaptadj: bool = False,
        num_nodes: int = 207, 
        adaptembed: int = 10
    ) -> None:
        super(GNN, self).__init__()

        modules = []
        in_size = in_feat
        for _ in range(n_layers): 
            modules.append(GNN_layer(in_size, hidden_size, adaptadj))
            in_size = 2 * hidden_size
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
        
    def pred_step(self, g, h): 
        """Encoded prediction used for the Actor Critic"""
        h = ptu.from_numpy(h)
        with torch.no_grad(): 
            if self.adaptadj: 
                g = self._update_edge(g)

            # h = h[..., 0]
            shape = h.shape
            # h = h.transpose(1,2).flatten(0,1)
            h = h.flatten(0,1)
            h = self.model(g, h)
            # h = h.reshape(shape[0], shape[2], shape[1]).transpose(1,2)
            h = h.reshape(*shape[:2], -1)
        return ptu.to_numpy(h)

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


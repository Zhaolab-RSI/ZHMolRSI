# HIL_zhuo_gnn.py
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=0, act_fn=nn.LeakyReLU()):
        super(GNNLayer, self).__init__(aggr='add')  # sum aggregation
        self.edge_dim = edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            act_fn,
            nn.Linear(out_channels, out_channels),
            act_fn
        )

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is None:
            m = torch.cat([x_i, x_j], dim=-1)
        else:
            m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(m)


class GNN_complex(nn.Module):
    def __init__(self, hid_dim, edge_dim, n_layers=4, attention=False, normalize=False, tanh=False):
        super(GNN_complex, self).__init__()
        self.hid_dim = hid_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers

        layers = []
        for _ in range(n_layers):
            layers.append(GNNLayer(hid_dim, hid_dim, edge_dim=edge_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, data_complex):
        x = data_complex.x
        edge_index = data_complex.edge_index
        edge_attr = data_complex.edge_attr

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


# ConBAP_zhuo_gnn.py
# %%
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from HIL import GNN_complex 
import numpy as np

class GNN(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()


        self.lin_node_complex = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.SiLU()
        )


        self.gnn = GNN_complex(hidden_dim, edge_dim=4, n_layers=4, attention=False)

        self.g = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
        )


        self.kl_loss = nn.KLDivLoss(reduction='sum')

    def forward(self, data_list):
        embeddings = []
        for data in data_list:
         
            data.x = self.lin_node_complex(data.x)

       
            x_c_n = self.gnn(data)

         
            x_c_n = global_mean_pool(x_c_n, data.batch)

            embeddings.append(x_c_n)

    
        embeddings_tensor = torch.stack(embeddings)
        return embeddings_tensor


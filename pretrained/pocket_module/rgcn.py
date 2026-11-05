from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling, GlobalAttentionPooling
from dgl import mean_nodes
from dgl.nn.pytorch.conv import RelGraphConv


class Embedder(nn.Module):

    def __init__(self, dims, num_rels=7, num_bases=-1):
        super(Embedder, self).__init__()
        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.layers = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = (*self.dims[-2:],)

        i2h = self.build_hidden_layer(47, self.dims[0])
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = self.build_hidden_layer(dim_in, dim_out)
            layers.append(h2h)

        h2o = self.build_output_layer(last_hidden, last)
        layers.append(h2o)

        return layers

    @property
    def current_device(self):
        return next(self.parameters()).device

    def build_hidden_layer(self, in_dim, out_dim):
        return RelGraphConv(
            in_dim,
            out_dim,
            self.num_rels,
            num_bases=self.num_bases,
            activation=F.relu
        )

    def build_output_layer(self, in_dim, out_dim):
        return RelGraphConv(
            in_dim,
            out_dim,
            self.num_rels,
            num_bases=self.num_bases,
            activation=None
        )

    def forward(self, g):
        h = g.ndata['feat'].to(self.current_device)

        for i, layer in enumerate(self.layers):
            h = layer(g, h, g.edata['type'])

        g.ndata['h'] = h
        del h

        embeddings = g.ndata.pop('h')

        return embeddings


class Model(nn.Module):

    def __init__(
        self,
        dims,
        device,
        num_rels,
        pool='att',
        num_bases=-1,
        pos_weight=0,
        clustered=False,
        num_clusts=8
    ):
        super(Model, self).__init__()

        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.pos_weight = pos_weight
        self.device = device
        self.clustered = clustered
        self.num_clusts = num_clusts

        self.embedder = Embedder(
            dims=dims,
            num_rels=num_rels,
            num_bases=num_bases
        )

    def forward(self, g):
        embeddings = self.embedder(g)
        return embeddings

    def rec_loss(self, embeddings, target_K, similarity=True):

        if similarity:
            K_predict = self.matrix_cosine(
                embeddings,
                embeddings
            )

        else:
            K_predict = self.matrix_dist(embeddings)
            target_K = torch.ones(
                target_K.shape,
                device=target_K.device
            ) - target_K

        reconstruction_loss = torch.nn.MSELoss()(
            K_predict,
            target_K
        )

        return reconstruction_loss

    @staticmethod
    def matrix_dist(a, plus_one=False):

        if plus_one:
            return torch.norm(
                a[:, None] - a,
                dim=2,
                p=2
            ) + 1

        return torch.norm(
            a[:, None] - a,
            dim=2,
            p=2
        )

    @staticmethod
    def matrix_cosine(a, b, eps=1e-8):

        a_n, b_n = (
            a.norm(dim=1)[:, None],
            b.norm(dim=1)[:, None]
        )

        a_norm = a / torch.max(
            a_n,
            eps * torch.ones_like(a_n)
        )

        b_norm = b / torch.max(
            b_n,
            eps * torch.ones_like(b_n)
        )

        sim_mt = torch.mm(
            a_norm,
            b_norm.transpose(0, 1)
        )

        return sim_mt

    def compute_loss(
        self,
        embeddings,
        target_K,
        rec_lam=1,
        similarity=True
    ):

        loss = rec_lam * self.rec_loss(
            embeddings,
            target_K,
            similarity=similarity
        )

        return loss

    def draw_rec(self, true_K, predicted_K, title=''):

        import matplotlib.pyplot as plt
        import seaborn as sns

        kws = {
            'cbar_kws': {'shrink': 0.5},
            'square': True,
            'vmin': 0,
            'vmax': 1
        }

        fig, (ax1, ax2) = plt.subplots(1, 2)

        sns.heatmap(
            true_K.clone().detach().numpy(),
            ax=ax1,
            **kws
        )

        sns.heatmap(
            predicted_K.clone().detach().numpy(),
            ax=ax2,
            **kws
        )

        ax1.set_title("Ground Truth")
        ax2.set_title("GCN")

        fig.suptitle(title)

        plt.tight_layout()
        plt.show()

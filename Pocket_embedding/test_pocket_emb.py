import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import torch
import numpy as np
import dgl
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import math
from collections import Counter
from rgcn import Model
from dgl.nn import AvgPooling
import csv



# ======== collate_fn ===========
def collate_fn(samples):
    graphs, names = zip(*samples)
    batched_graph = dgl.batch(graphs)
    return batched_graph, names
    
    
    
    
class GraphDataset(Dataset):


    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        graph, _, _ = pickle.load(open(os.path.join(self.path, self.all_graphs[idx]), 'rb'))
        graph = nx.to_undirected(graph)
        edge_attrs = {edge: self.edge_map[label] for edge, label in nx.get_edge_attributes(graph, 'label').items()}
        nx.set_edge_attributes(graph, name='type', values=edge_attrs)

        for node in graph.nodes():
            d = graph.nodes[node]
            for k, v in d.items():
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    raise ValueError(f"Node {node} has invalid attr {k} in {self.all_graphs[idx]}")
            attrs = []
            attrs.extend([d.get(k, 0.0) for k in ['ssZp', 'Dp', 'suiteness']])
            attrs.extend([v for k, v in d.items() if k.startswith('sin_') or k.startswith('cos_')])
            attrs.extend(d.get('dbn_onehot', [0.0, 0.0]))
            attrs.extend(d.get('nuc_onehot', [0.0] * 4))
            attrs.extend(d.get('bin_onehot', [0.0] * 14))
            graph.nodes[node]['feat'] = torch.tensor(attrs, dtype=torch.float32)
        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, node_attrs=['feat'], edge_attrs=['type'])
        g_dgl.title = self.all_graphs[idx]
        return g_dgl, self.all_graphs[idx]
    def __init__(self, annotated_path):
        self.path = annotated_path
        self.all_graphs = sorted([f for f in os.listdir(annotated_path) if f.endswith('.p')])
        self.edge_map = {
    'B53': 0,
    'cHS': 1,
    'cWH': 2,
    'cWS': 3,
    'tHS': 4,
    'tWH': 5,
    'tWS': 6}
        self.n = len(self.all_graphs)
    def _get_edge_data(self):
        edge_counts = Counter()
        for fname in self.all_graphs:
            g, _, _ = pickle.load(open(os.path.join(self.path, fname), 'rb'))
            labels = {edata['label'] for _, _, edata in g.edges(data=True)}
            edge_counts.update(labels)
        edge_map = {label: i for i, label in enumerate(sorted(edge_counts))}
        return edge_map, edge_counts

def extract_embeddings(model_path, graph_dir, output_dir, out_csv, device='cpu'):
    device = torch.device(device)
    model_dict = torch.load(model_path, map_location=device)
    model = Model(dims=[16,16,16], device=device, num_rels=7)
    model.load_state_dict(model_dict['model_state_dict'])
    model = model.to(device)
    model.eval()

    dataset = GraphDataset(graph_dir)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    all_embeds = []
    all_ids = []
    avg_pool = AvgPooling()

    with open(os.path.join(output_dir, out_csv), 'w', newline='') as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(['pdb_id'] + [f'emb_{i}' for i in range(16)])
         
         for batch_graph, names in tqdm(dataloader):
             batch_graph = batch_graph.to(device)
             with torch.no_grad():
                  node_emb= model(batch_graph)

                  graph_emb = avg_pool(batch_graph, node_emb)
                
                  embedding = graph_emb.squeeze(0)
                  all_embeds.append(embedding.cpu().numpy())
                  all_ids.extend(names)
                  for name in names:
                      writer.writerow([name] + embedding.tolist())




if __name__ == '__main__':

########Provide your actual path.####################
    extract_embeddings(
        model_path = './pretrained_pocket_model/pretrained_pocket_model.pth',
        graph_dir = '/ZHMolRSI/example/6HAG/pocket_emb/nx_p',


        output_dir = '/ZHMolRSI/example/6HAG/pocket_emb/model_pocket_emb_out',
        out_csv='6HAG_poc_emb.csv',
        device = 'cpu'
    )


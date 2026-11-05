import pickle
import sys
import os
from collections import Counter
import itertools
import math

if __name__ == '__main__':
    sys.path.append('../')

import networkx as nx
from tqdm import tqdm
import dgl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

from data_processor_zhuo.node_sim import SimFunctionNode, k_block_list 

class V1(Dataset):       
    def __init__(self, sim_function="R_1",        
                    annotated_path='./data/out_dump_500/',
                    get_sim_mat=True,
                    depth=4,
                    shuffle=False,   
                    seed=0,         
                    clustered=False, 
                    num_clusters=8):  #
        """
            Setup for data loader.

            Arguments:    
                sim_function (str): which node kernel to use (default='R1', see `node_sim.py`).  #用于设置节点相似性计算的函数类型（默认使用 "R_1"）。
                annotated_path (str): path to annotated graphs (see `annotator.py`).
                get_sim_mat (bool): whether to compute a node similarity matrix (deault=True).  #get_sim_mat: 布尔值，指示是否计算节点相似性矩阵。
                nucs (bool): whether to include nucleotide ID in node (default=False).  # 是否包含核苷酸 ID（用于节点特征）
                depth (int): number of hops to use in the node kernel.  #图的节点核的深度。
        """
        print(f">>> fetching data from {annotated_path}")
        self.path = annotated_path
        self.all_graphs = sorted(os.listdir(annotated_path))
        if seed:
            print(f">>> shuffling with random seed {seed}")
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.all_graphs)
        #build edge map
        self.edge_map, self.edge_freqs = self._get_edge_data()  
        self.num_edge_types = len(self.edge_map)
        self.clustered = clustered
        self.num_clusters = num_clusters

            
        print(f">>> found {self.num_edge_types} edge types, frequencies: {self.edge_freqs}")

        self.node_sim_func = SimFunctionNode(method=sim_function, IDF=self.edge_freqs, depth=depth) 

        self.n = len(self.all_graphs)
        
        self.get_sim_mat = get_sim_mat 


    def __len__(self):
        return self.n    
   # def get_node_features(H, node):

       # return np.array(node_attrs, dtype=np.float32)

    def __getitem__(self, idx):  
        """
            Returns one training item at index `idx`.
        """
        graph, _, ring = pickle.load(open(os.path.join(self.path, self.all_graphs[idx]), 'rb'))
        file_name_RNA = self.all_graphs[idx] 
        print (f"第 {idx} 个结构名字是{file_name_RNA}")                
 
        

       
        graph = nx.to_undirected(graph)       
            
           # print(f"Edge: ({n1}, {n2}), Attributes: {attributes}")


        type_attrs = {edge: self.edge_map[label] for edge, label in nx.get_edge_attributes(graph, 'label').items()}
        nx.set_edge_attributes(graph, name='type', values=type_attrs)
       # for n1, n2, attributes in graph.edges(data=True):
           # print(f"Edge: ({n1}, {n2}), Attributes: {attributes}")
        ##构造节点特征
        node_list = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        for n, d in graph.nodes(data=True):
            for k, v in d.items():
                if v is None:
                   print(f"节点 {n} 的属性 '{k}' 是 None")
                   raise ValueError(f"❌节点 {n} 的属性 {k}为None程序终止。")  
                elif isinstance(v, float) and math.isnan(v):
                   print(f"节点 {n} 的属性 '{k}' 是 NaN")
                   raise ValueError(f"❌节点 {n} 的属性 {k}为Nan程序终止。")
        for node in node_list:
            node_attrs = []
            node_data = graph.nodes[node]
            keys_continuous = ['ssZp', 'Dp', 'suiteness']
            node_attrs.extend([node_data.get(k, 0.0) for k in keys_continuous])
            angle_features = [v for k, v in node_data.items() if k.startswith('sin_') or k.startswith('cos_')]
            node_attrs.extend(angle_features)
            node_attrs.extend(node_data.get('dbn_onehot', [0.0, 0.0]))
            node_attrs.extend(node_data.get('nuc_onehot', [0.0]*4))
            node_attrs.extend(node_data.get('bin_onehot', [0.0]*14))
            node_attrs_array = np.array(node_attrs, dtype=np.float32)
            graph.nodes[node]['feat'] = torch.tensor(node_attrs_array, dtype=torch.float)
        #for node in graph.nodes:
           # print(f"Node: {node}, feat: {graph.nodes[node]['feat']}")
            


        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph,  node_attrs=['feat'],edge_attrs=['type'])
        feat = g_dgl.ndata['feat']
        #has_nan = torch.isnan(feat).any()
        #has_inf = torch.isinf(feat).any()
        #print("Feature has NaN:", has_nan)
        #print("Feature has Inf:", has_inf)
        g_dgl.title = self.all_graphs[idx]            
     
        if self.get_sim_mat:
            # put the rings in same order as the dgl graph
            ring = dict(sorted(ring.items()))
            #print("Node data:", g_dgl.ndata)
           # print("Edge data:", g_dgl.edata)
            return g_dgl, ring, [idx]

        else:
            return g_dgl, [idx]   

    def _get_edge_data(self):
        """
            Get edge type statistics, and edge map.
        """
        edge_counts = Counter()
        edge_labels = set()
        print("Collecting edge data...")
        graphlist = os.listdir(self.path)
        for g in tqdm(graphlist):
            graph,_,_ = pickle.load(open(os.path.join(self.path, g), 'rb'))
            if len(graph.nodes()) < 4:
                print(len(graph.nodes()), g)
            # assert len(graph.nodes()) > 0, f"{len(graph.nodes())}"
            edges = {e_dict['label'] for _,_,e_dict in graph.edges(data=True)}
            edge_counts.update(edges)
        
        edge_map = {label:i for i,label in enumerate(sorted(edge_counts))}
        IDF = {k: np.log(len(graphlist)/ edge_counts[k] + 1) for k in edge_counts}
        #print (IDF)
        #print (edge_map)
        return edge_map, IDF       

def collate_wrapper(node_sim_func, get_sim_mat=True):
    """
        Wrapper for collate function so we can use different node similarities.
    """
    if get_sim_mat:  
        def collate_block(samples):   
            graphs, rings, idx  = map(list, zip(*samples))   
            #print (rings)
           
           # print (idx)
            idx = np.array(idx)
            batched_graph = dgl.batch(graphs)
            K = k_block_list(rings, node_sim_func)   
            return batched_graph, torch.from_numpy(K).float(), torch.from_numpy(idx)
    else:   
        def collate_block(samples):  
            # The input `samples` is a list of pairs
            #  (graph, label).
            # print(len(samples))
            graphs,idx = map(list, zip(*samples))
            idx = np.array(idx)
            batched_graph = dgl.batch(graphs)
           
            return batched_graph, [1 for _ in samples],  torch.from_numpy(idx)
    return collate_block

class Loader():  
    def __init__(self,
                 annotated_path='./data/out_dump_500/',
                 batch_size=128,  
                 num_workers=20,
                 sim_function="R_1",  
                 shuffle=False,
                 seed=0,    
                 get_sim_mat=True,   
                 depth=4):   
        """
        Wrapper class to call with all arguments and that returns appropriate data_loaders
        :param pocket_path:
        :param ligand_path:
        :param batch_size:    #每个批次的样本数。
        :param num_workers:   #数据加载时使用的线程数。
        :param augment_flips: perform numpy flips
        :param ram: store whole thing in RAM
        :param siamese: for the batch siamese technique
        :param full_siamese for the true siamese one
        """
        self.all_graphs = sorted(os.listdir(annotated_path))
        self.batch_size = batch_size     #保存批次大小和工作线程数。
    
        self.num_workers = num_workers
        self.dataset = V1(annotated_path=annotated_path,
                          sim_function=sim_function,
                          get_sim_mat=get_sim_mat,
                          shuffle=shuffle,
                          seed=seed,
                          depth=depth)

        self.num_edge_types = self.dataset.num_edge_types   

    def get_data(self, k_fold=0):
        n = len(self.dataset)   
        indices = list(range(n))
        collate_block = collate_wrapper(self.dataset.node_sim_func, self.dataset.get_sim_mat) #使用 collate_wrapper 创建一个合适的合并函数。
       
        if k_fold > 1:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=k_fold)
            for train_indices, test_indices in kf.split(np.array(indices), np.array(indices)):
                train_set = Subset(self.dataset, train_indices)
                test_set = Subset(self.dataset, test_indices)

                train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                          num_workers=self.num_workers, collate_fn=collate_block)
                test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                         num_workers=self.num_workers, collate_fn=collate_block)

                yield train_loader, test_loader

        #如果不使用交叉验证，按比例将数据集划分为训练集和测试集。
        else:
            split_train, split_valid = 0.8, 0.2      #设置训练集和验证集的比例（此处都为 80%）。
            train_index, valid_index = int(split_train * n), int(split_valid * n)

            train_indices = indices[:train_index]    #前80%
            test_indices = indices[train_index:]     #剩下20%

            train_set = Subset(self.dataset, train_indices)
            test_set = Subset(self.dataset, test_indices)

            print("training graphs ", len(train_set))
            print("testing graphs ", len(test_set))

            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)

            # return train_loader, valid_loader, test_loader
            yield train_loader, test_loader
        #full loader


#推理阶段（Inference Phase）通常指的是模型训练完成后，用来进行模型评估或实际应用的阶段。与训练阶段不同，推理阶段的目标是通过输入新的数据（例如，未见过的测试集、实时数据或生产环境中的数据），得到模型的预测结果，而不是用来更新模型的参数。推理阶段通过使用测试集（或验证集）来评估模型的性能，如精度（accuracy）、召回率（recall）、F1 分数等指标。
class InferenceLoader(Loader):    
    def __init__(self,
                 annotated_path,   #数据的路径
                 batch_size=5,
                 num_workers=20):
        super().__init__(
            annotated_path=annotated_path,
            batch_size=batch_size,
            num_workers=num_workers)
        self.dataset.all_graphs = sorted(os.listdir(annotated_path))

    def get_data(self):
        collate_block = collate_wrapper(None, get_sim_mat=False)
        train_loader = DataLoader(dataset=self.dataset, shuffle=False, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collate_block)
        return train_loader
if __name__ == '__main__':
    loader = Loader(shuffle=False,seed=99, batch_size=1, num_workers=1)
    data = loader.get_data(k_fold=5)
    for train, test in data:
        print(len(train), len(test))

    pass

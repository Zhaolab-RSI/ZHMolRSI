# %%
import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import pickle
from scipy.spatial import distance_matrix
import multiprocessing             
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
#from utils import *
import warnings
from threading import Lock
my_lock = Lock()      
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
   
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) 
 
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)
   
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))


def get_edge_index(mol, graph):
    edge_features = []
    for bond in mol.GetBonds():    
        i = bond.GetBeginAtomIdx()   
        j = bond.GetEndAtomIdx()
        feats = bond_features(bond)
        graph.add_edge(i, j)
        edge_features.append(feats)
        edge_features.append(feats)

    return torch.stack(edge_features)

def bond_features(bond):
   
    bt = bond.GetBondType() 
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(),bond.GetIsConjugated()]
    return torch.Tensor(fbond)


def mol2graph(mol,pdbid):   
    try: 
       graph = nx.Graph()
       atom_features(mol, graph)
       edge_features = get_edge_index(mol, graph)
       graph = graph.to_directed()   
       x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
       edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

       return x, edge_index, edge_features
    except Exception as e:
        
        
        with open("failed_molecules_1.txt", "a") as f:      
            f.write(pdbid + "\n")
        print(f"[Error] Skipped molecule: {Chem.MolToSmiles(mol)}. Reason: {e}")
        return  (None, None, None) 



def inter_graph(ligand, pocket, dis_threshold=10.):
    atom_num_l = ligand.GetNumAtoms()   
    atom_num_p = pocket.GetNumAtoms()
    #print (dis_threshold)
    graph_inter = nx.Graph()
    pos_l = ligand.GetConformer().GetPositions()   
    pos_p = pocket.GetConformer().GetPositions()


    dis_matrix_l = distance_matrix(pos_l, pos_l)
    dis_matrix_p = distance_matrix(pos_p, pos_p)
    dis_matrix_lp = distance_matrix(pos_l, pos_p)

    node_idx_l = np.where(dis_matrix_l < dis_threshold)
    for i, j in zip(node_idx_l[0], node_idx_l[1]):
        graph_inter.add_edge(i, j, feats=torch.tensor([1, 0, 0, dis_matrix_l[i, j]]))

    node_idx_p = np.where(dis_matrix_p < dis_threshold)
    for i, j in zip(node_idx_p[0], node_idx_p[1]):
        graph_inter.add_edge(i + atom_num_l, j + atom_num_l, feats=torch.tensor([0, 1, 0, dis_matrix_p[i, j]]))

    node_idx_lp = np.where(dis_matrix_lp < dis_threshold)
    for i, j in zip(node_idx_lp[0], node_idx_lp[1]):
        graph_inter.add_edge(i, j + atom_num_l, feats=torch.tensor([0, 0, 1, dis_matrix_lp[i, j]]))

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v, _ in graph_inter.edges(data=True)]).T
    edge_attrs_inter = torch.stack([feats['feats'] for _, _, feats in graph_inter.edges(data=True)]).float()
    #print (edge_index_inter)
    #print (edge_attrs_inter)
    df_edge_index = pd.DataFrame(edge_index_inter.numpy().T, columns=['Source Node', 'Target Node'])
    df_edge_attrs = pd.DataFrame(edge_attrs_inter.numpy(), columns=['Feats 1', 'Feats 2', 'Feats 3', 'Distance'])
    #df_edge_index.to_csv('edge_index.csv', index=False)
   # df_edge_attrs.to_csv('edge_attrs.csv', index=False)

    return edge_index_inter, edge_attrs_inter



# %%
def mols2graphs(complex_path, pdbid, save_path_complex, pocket_dis = 10):
    #print(pdbid)
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    x_l, edge_index_l,edge_features_l = mol2graph(ligand,pdbid)  
    x_p, edge_index_p,edge_features_p = mol2graph(pocket,pdbid)  
    if   edge_index_l is None:
         return
    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x = torch.cat([x_l, x_p], dim=0)   
    edge_index_inter, edge_attrs_inter = inter_graph(ligand, pocket, dis_threshold=10)  
    pos = torch.concat([pos_l, pos_p], dim=0)
    parser = PDBParser(QUIET=True)
    
    
   

    c_size_l = atom_num_l
    c_size_p = atom_num_p

    c_size_complex = c_size_l + c_size_p
 
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
     
    data_complex =Data(x=x, edge_attr=edge_attrs_inter, edge_index=edge_index_inter, pos=pos)   
    data_complex.__setitem__('c_size', torch.LongTensor([c_size_complex]))  

    torch.save(data_complex, save_path_complex)
    # return data



class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=10, graph_type='ConBAP', num_process=4, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))
        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_l_list = []
        graph_path_p_list = []
        graph_path_aa_list = []
        graph_path_complex_list = []
        for i, row in data_df.iterrows():
            cid = row['pdb']
            if type(cid) != str:
                cid = str(int(cid))
            complex_dir = os.path.join(data_dir, cid)

            graph_path_complex = os.path.join(complex_dir, f"{graph_type}-{cid}_complex_{self.dis_threshold}A.pyg")  
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")
            
            complex_path_list.append(complex_path)
            complex_id_list.append(cid)

            graph_path_complex_list.append(graph_path_complex)

        if self.create:
            print('Generate complex graph...')
            #multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, complex_id_list,graph_path_complex_list, dis_thresholds))
            pool.close()
            pool.join()

        self.complex_ids = complex_id_list  
        #['1q8n_A_MGR_1', '1q8n_A_MGR_2', '2au4_A_GTP_1', '2au4_A_GTP_2', '4lw0_B_ADE_1', '4lw0_B_ADE_2']
        
        self.graph_paths_complex = graph_path_complex_list
     
    def __len__(self):  
       return len(self.graph_paths_complex)
   

    def __getitem__(self, idx):    
       #print (idx)
       #print(self.graph_paths_complex[idx])
       return torch.load(self.graph_paths_complex[idx])

        
class PLIDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.custom_collate)
    @staticmethod
    def custom_collate(batch):
        imgL = [item[0] for item in batch]
        imgR = [item[1] for item in batch] 
        return imgL, imgR
        
        
class ToyGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list  

    def __len__(self):
        return len(self.data) // 2  

    def __getitem__(self, idx):
       
        return self.data[2 * idx], self.data[2 * idx + 1]
if __name__ == '__main__':
    data_root = './zhuo_data/zhuo_2/put1'
    data_root_1 = './zhuo_data/zhuo_2'
    data_dir = os.path.join(data_root)
    data_df = pd.read_csv(os.path.join(data_root_1, 'put_1.csv'))
    device = torch.device('cuda:0')

    toy_set = GraphDataset(data_dir, data_df, graph_type='ConBAP', dis_threshold=10, create=True)

    train_loader=ToyGraphDataset(toy_set)
    train_data = PLIDataLoader(train_loader, batch_size=1, shuffle=True, num_workers=0)
    for batch_idx, (imgL, imgR) in enumerate(train_data): 
 
      
        print("  imgL:", imgL)
        print("  imgR:", imgR)
    print('finish!')

    


# %%

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

import config
from model import GNN_inter
from tqdm import tqdm
my_lock = Lock()     
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')


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
            #complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")
            
            #complex_path_list.append(complex_path)
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
       print (idx)
       print(self.graph_paths_complex[idx])
       return torch.load(self.graph_paths_complex[idx])

        
class PLIDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        
class ToyGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list 

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
      
        return self.data[2 * idx], self.data[2 * idx + 1]
if __name__ == '__main__':

    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current device:", DEVICE)
    model_path = './pretrained_interface_model/pretrained_interface_model.pth'  

#######Provide your actual path.####################
    data_root = '/ZHMolRSI/example/6HAG/interact_emb/out'
    data_root_1 = '/ZHMolRSI/example/6HAG/interact_emb'
    data_dir = os.path.join(data_root)
    data_df = pd.read_csv(os.path.join(data_root_1, '6HAG.csv'))


    graph_dataset = GraphDataset(data_dir, data_df, graph_type='ZHMolRSI', dis_threshold=10, create=False)


    model = GNN_inter(33,128).to(DEVICE)                  
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()


    embeddings = []
    complex_ids = []

    with torch.no_grad():
        for idx in tqdm(range(len(graph_dataset))):
            data = graph_dataset[idx].to(DEVICE) 
            emb = model([data])  
            emb = emb.cpu().numpy().squeeze() 
            embeddings.append(emb)
            complex_ids.append(graph_dataset.complex_ids[idx])
  
##############Provide your actual path.#########################
    output_dir = '/ZHMolRSI/example/6HAG/interact_emb/model_interact_emb_out'

    output_file_3 ='6HAG_inter_emb.csv'

    
    embedding_array = np.array(embeddings)
    df = pd.DataFrame(embedding_array, columns=[f"emb_{i}" for i in range(embedding_array.shape[1])])
    df.insert(0, "pdb_id", complex_ids)
    df.to_csv(os.path.join(output_dir, output_file_3), index=False)



# %%

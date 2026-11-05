import os
import sys
import pickle
import multiprocessing

import numpy as np

import pandas as pd
from Bio.PDB import *

def get_all_rna_residues(RNA_df):
    """
    Function to get all RNA residues in the structure.
    
    Arguments:
        struct (Bio.PDB.Structure): BioPython object for full structure.

    Returns:
        rna_residues (list): List of all RNA residues in the structure.
    """
    #print (RNA_df)
    #print (struct)
    rna_residues = []
    RNA_df = RNA_df.drop_duplicates(subset=['chain', 'res_name', 'unique_id'])
    #print (res2)
    for RNA in RNA_df.itertuples(): 
        #print(RNA)
        
        res =RNA.res_name
        #print ("111")
        #print (res)
            
        if res in ['A', 'U', 'C', 'G']:  # RNA nucleotides
               
           res_info = f"{RNA.chain}:{RNA.res_name}:{RNA.unique_id}"
           #print (res_info)
           rna_residues.append(res_info)
               #print (res_info)
    #print (rna_residues)
    return rna_residues

def full_struc_analyse(strucpath):
    """
    Analyses a full PDB structure for its RNA residues.

    Arguments:
        strucpath (str): Path to PDB structure.

    Returns:
        rna_residues_info (list): List of RNA residue information for the entire structure.
    """
    try:
        struc_dict = MMCIF2Dict.MMCIF2Dict(strucpath)  
        parser = MMCIFParser(QUIET=True)
        pdbstruc = parser.get_structure("", strucpath)
    except Exception as e:
        print(f"Structure loading error {e}")
        return None 

    RNA_dict = {}
    try:
        RNA_dict['position'] = struc_dict['_atom_site.id']     
        RNA_dict['res_name'] = struc_dict['_atom_site.label_comp_id']    
        RNA_dict['chain'] = struc_dict['_atom_site.auth_asym_id']       
        RNA_dict['unique_id'] = struc_dict['_atom_site.label_seq_id']   
    except:
        print("Ligand not detected.")
        return None    

    num_models = len(pdbstruc)  
    model = pdbstruc[0]
    
    try:
        RNA_df = pd.DataFrame.from_dict(RNA_dict)   
    except ValueError:
        RNA_df = pd.DataFrame(RNA_dict, index=[0])

    RNA_df['position'] = pd.to_numeric(RNA_df['position'])
    pdbid = os.path.basename(strucpath).split(".")[0]
    RNA_df['pdbid'] = pdbid
    #print (RNA_df)
   
    rna_residues = get_all_rna_residues(RNA_df)
    
    return rna_residues, strucpath

def binding_wrapper(strucpath):
    try:
        return full_struc_analyse(strucpath)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(f"Failed on {strucpath} :: {e}")
        return None

def process_all(pdb_path, dump_path, restart=None, parallel=False):
    """
    Main function for building lig_dict for RNA residues.

    Arguments:
        pdb_path (str): path to PDBs
        dump_path (str): file path to write the RNA residue information.
        restart (str): path to dictionary to start from.
        parallel (bool): If True runs with multiprocessing
    """
    pdbs = [os.path.join(pdb_path, p) for p in os.listdir(pdb_path)]   
    num_pdbs = len(pdbs)
    done = []
    if not restart is None:
        try:
            r = pickle.load(open(restart, 'rb'))
            done = list(r.keys())
        except FileNotFoundError:
            print(f"Could not load dict {restart}")

    RNA_ = {}

    todo = iter([p for p in pdbs if p not in done])
    num_done = 0

    pool = multiprocessing.Pool(processes=20)
    for result in pool.imap_unordered(full_struc_analyse, todo):
        if result is None:
            print("failed")
            continue
        rna_residues, p = result
        num_done += 1
        print(f"done {p}, {num_done} of {num_pdbs}")  
        RNA_[os.path.basename(p)] = rna_residues
      
        for pdb_file, rna_residues in RNA_.items():
            print(f"{pdb_file}: {rna_residues}")
        pickle.dump(RNA_, open(dump_path, 'wb')) 
    return RNA_

if __name__ == "__main__":
    pdb_path = os.path.join("/media/zc/Elements SE/新数据集/ZHMolRSI数据集/网站/example/2Z74/pocket_emb/bind_site_cif")  # 你的cif文件路径
    RNA_=process_all(pdb_path, "/media/zc/Elements SE/新数据集/ZHMolRSI数据集/网站/example/2Z74/pocket_emb/1/1.p")  # 输出的.p文件路径

    

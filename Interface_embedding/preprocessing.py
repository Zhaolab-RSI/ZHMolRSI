# %%
import os
import pickle
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rdkit import RDLogger
from pymol import cmd
RDLogger.DisableLog('rdApp.*')


def load_first_valid_mol(sdf_path):
    
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=True)
    for mol in suppl:
        if mol is not None:
            return mol
    print(f"Unable to process ligand from: {sdf_path}")
    return None


def generate_pocket_and_rdkit(pdb_id, ligand_path, rna_path, output_dir, distance=10):
    


    ligand_name = os.path.splitext(os.path.basename(ligand_path))[0]
    ligand_out_dir = os.path.join(output_dir, ligand_name)
    os.makedirs(ligand_out_dir, exist_ok=True)

    pocket_pdb_path = os.path.join(ligand_out_dir, f'{ligand_name}_Pocket_{distance}A.pdb')
    rdkit_path = os.path.join(ligand_out_dir, f'{ligand_name}_{distance}A.rdkit')

 
    if os.path.exists(rdkit_path) and os.path.exists(pocket_pdb_path):
        return None

    try:
       
        cmd.delete('all')
        cmd.load(rna_path, 'rna')
        cmd.remove('rna and hydro')
        cmd.remove('resn HOH')

        cmd.load(ligand_path, 'lig')
        cmd.remove('lig and hydro')

    
        cmd.select('Pocket', f'byres rna within {distance} of lig')
        cmd.save(pocket_pdb_path, 'Pocket')
        cmd.delete('all')
    except Exception as e:
        return f"Ligand pocket generation error: {ligand_path}, {e}"

   
    ligand = load_first_valid_mol(ligand_path)
    if ligand is None:
        return f"Ligand load error: {ligand_path}"

 
    pocket = Chem.MolFromPDBFile(pocket_pdb_path, sanitize=False, removeHs=True)
    if pocket is None:
        return f"Pocket load error: {pocket_pdb_path}"

  
    try:
        with open(rdkit_path, 'wb') as f:
            pickle.dump((ligand, pocket), f)
    except Exception as e:
        return f"RDKit save error: {rdkit_path}, {e}"

    return None  


def process_single_pdb(pdb_id, ligand_dir, rna_dir, out_dir, distance=10):

    A = pdb_id.split("_lig")[0]
    rna_path = os.path.join(rna_dir, f'{A}_RNA.pdb')
    ligands_subdir = os.path.join(ligand_dir, pdb_id)
    if not os.path.exists(ligands_subdir):
        return [f"No ligand folder for {pdb_id}"]

    sdf_files = sorted([os.path.join(ligands_subdir, f) for f in os.listdir(ligands_subdir) if f.endswith('.sd')])
    errors = []

    for sdf in sdf_files:
        err = generate_pocket_and_rdkit(pdb_id, sdf, rna_path, out_dir, distance)
        if err:
            errors.append(err)

    return errors


def main(data_root, distance=10, max_workers_outer=4):
    ligand_dir = os.path.join(data_root, 'ligand_sd')
    rna_dir = os.path.join(data_root, 'RNA')
    out_dir = os.path.join(data_root, 'out')

    pdb_ids = [name for name in os.listdir(ligand_dir) if os.path.isdir(os.path.join(ligand_dir, name))]
    all_errors = []

    with ProcessPoolExecutor(max_workers=max_workers_outer) as executor:
        futures = {executor.submit(process_single_pdb, pdb_id, ligand_dir, rna_dir, out_dir, distance): pdb_id for pdb_id in pdb_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing all PDBs'):
            pdb_id = futures[future]
            try:
                errors = future.result()
                if errors:
                    all_errors.extend(errors)
            except Exception as e:
                all_errors.append(f"Process error for {pdb_id}: {e}")


    error_file = os.path.join(data_root, 'processing_errors1.txt') 
    if all_errors:
        with open(error_file, 'w') as f:
            for err in all_errors:
                f.write(err + '\n')
        print(f"Errors recorded in {error_file}")
    else:
        print("All files processed successfully, no errors.")


if __name__ == '__main__':

####Provide your actual path#####
    data_root = '/ZHMolRSI/example/6HAG/interact_emb'
    distance = 10
    max_workers_outer = 10  

    main(data_root, distance, max_workers_outer)


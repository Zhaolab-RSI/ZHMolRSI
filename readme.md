# ZHMolRSI

## Overview of ZHMolRSI

**Pocket-Aware Hierarchical Pretrained Representations and Capsule Fusion for RNA-Small Molecule Binding Prediction**

ZHMolRSI is a hierarchical multimodal framework for RNA–small molecule binding prediction. It integrates pretrained RNA sequence representations, pocket representations, and ligand-dependent interface features through capsule-based fusion.

The framework supports two downstream prediction tasks:

- **ZHMolRSI_affi:** RNA–small molecule binding affinity prediction.
- **ZHMolRSI_dock:** RNA–small molecule docking pose scoring.

## Environment Setup

### Prerequisites

The following software environments are required:

- Conda
- Python 3.6.13 (Pocket module)
- Python 3.10 (Interface and Capsule fusion modules)
- PyTorch 1.9.1 (Pocket module)
- PyTorch 2.4.1 (Interface and Capsule fusion modules)
- CUDA 12.1-compatible environment for the provided CUDA-enabled extension wheels

### Installation Steps

Follow these steps to set up the environments. We have tested the following configurations in separate clean Conda environments.

**(1) Pocket Module**

Create and activate the Conda environment:

```bash
conda create -n RSI_p python=3.6.13 -y
conda activate RSI_p

pip install numpy==1.19.5

conda install pytorch=1.9.1 cpuonly \
    "mkl<2024.1" "intel-openmp<2024.1" \
    -c pytorch -y

pip install tensorboard==2.2.2
pip install dgl==0.4.3
pip install seaborn==0.10.1
pip install tqdm==4.64.1
```

**(2) Interface Module and Capsule Fusion Module**

Create and activate the Conda environment:

```bash
conda create -n RSI_i python=3.10 -y
conda activate RSI_i

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install pandas
pip install biopython
pip install scipy
conda install -c conda-forge rdkit -y
pip install torch_geometric

pip install pyg_lib-0.4.0+pt24cu121-cp310-cp310-linux_x86_64.whl
pip install torch_cluster-1.6.3+pt24cu121-cp310-cp310-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt24cu121-cp310-cp310-linux_x86_64.whl
pip install scikit-learn
pip install tensorboard
```

**Note:** The PyTorch Geometric extension wheels are built for Python 3.10, PyTorch 2.4.1, and CUDA 12.1 on Linux x86_64. Make sure the corresponding wheel files are available in your working directory before installation.

## Data & Pretrained Models

- **Data Files:** Available at [Zenodo](https://doi.org/10.5281/zenodo.23968600).
- **RNA Language Model:** [RiNALMo](https://github.com/lbcb-sci/RiNALMo).
- **RNA Structural Annotation:** [RNAglib](https://github.com/cgoliver/rnaglib).

## Workflow

The ZHMolRSI framework consists of three main representation modules and two downstream prediction tasks.

1. **Sequence Representation:** Extract RNA sequence embeddings using RiNALMo.
2. **Pocket Representation:** Generate RNA pocket embeddings using the pretrained pocket module.
3. **Interface Representation:** Generate RNA–small molecule interface embeddings using the pretrained interface module.
4. **Capsule-based Fusion:** Integrate sequence, pocket, and interface representations.
5. **Downstream Prediction:** Perform binding affinity prediction or docking pose scoring.

## Usage

### 1. Pretrained Embedding Preparation

**(1) Sequence Embedding**

- Generate RNA sequence embeddings using [RiNALMo](https://github.com/lbcb-sci/RiNALMo).

**(2) RNA Pocket Embedding**
- **6HAG Example:** All files required for the 6HAG example are available at [Zenodo](https://doi.org/10.5281/zenodo.23968600).
- Prepare RNA pocket structures for the RNA–small molecule complexes using a spatial distance cutoff of 10 Å.
- In `./example/6HAG/`, the `lig_pose` directory contains 100 decoy poses generated using XDock.
- The `./example/6HAG/pocket_emb/bind_site_cif/` directory stores the pocket structures in CIF format.
- Make sure that nucleotide residue numbers are positive rather than negative.
- Use [RNAglib](https://github.com/cgoliver/rnaglib) to obtain DSSR annotations and generate the corresponding pickle files in `./example/6HAG/pocket_emb/pickle/`.

Run the following scripts:

```bash
python ./Pocket_embedding/build_annotator/main_pocket_emb.py

python ./Pocket_embedding/test_pocket_emb.py
```

The RNA pocket embeddings will be saved to:

```text
./example/6HAG/pocket_emb/model_pocket_emb_out/6HAG_poc_emb.csv
```

**(3) Interface Embedding**

- Prepare the ligand structure in SD format (`ligand.sd`) and the RNA structure in PDB format (`RNA.pdb`).

Run the following scripts:

```bash
python ./Interface_embedding/preprocessing.py

python ./Interface_embedding/build_data.py

python ./Interface_embedding/test_inter_emb.py
```

The interface embeddings will be saved to:

```text
./example/6HAG/interact_emb/model_interact_emb_out/6HAG_inter_emb.csv
```

### 2. Docking Pose Scoring

Run the following script:

```bash
python ./Predict_docking_score/pre_dock_score.py
```

The prediction results will be saved in:

```text
./example/6HAG/prediction/
```

### 3. Binding Affinity Prediction

Run the following script:

```bash
python ./Predict_affinity/pre_affinity.py
```

### 4. Reproduction of ZHMolRSI Results

Scripts for reproducing the results of both prediction tasks are provided in:
- **Docking Score Test Set:** All files related to the `docking_score_test_set` are available at [Zenodo](https://doi.org/10.5281/zenodo.23968600).

```text
./predict_task/affinty_test/test_affi_r.py
./predict_task/docking_score_test/test_docking_score.py
```

### 5. Model Training

The training scripts are organized into the following directories:

- **Pocket Module and Interface Module:** `./pretrained/`
- **Capsule Fusion Module:** `./capsule_train/`

The corresponding training procedures and configurations are provided in these directories.

## Contact

For questions, comments, or suggestions regarding ZHMolRSI, please contact the authors.
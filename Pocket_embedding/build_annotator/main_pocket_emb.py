# -*- coding: utf-8 -*-
import os
import sys

from binding_pocket import process_all
from build_dataset import get_binding_site_graphs_all
from build_annotator import annotate_all


# =====================
# 总路径
# =====================
base_dir = "/media/zc/My Passport/卓晨/ZHMolRSI1.0/2.0/example/6HAG/pocket_emb"

# =====================
# 输入输出路径
# =====================

# 第一步输入：结合位点 cif 文件夹
bind_site_cif_dir = os.path.join(base_dir, "bind_site_cif")

# 第一步输出：RNA 残基信息 .p 文件
rna_info_dir = os.path.join(base_dir, "1")
rna_info_p = os.path.join(rna_info_dir, "1.p")

# 第二步输入：完整 RNA graph 的 pickle 文件夹
# 文件格式应该类似：
# 2Z74_lig_1_site.pickle
# 或者与你 1.p 中的 pdbid 对应
pickle_dir = os.path.join(base_dir, "pickle")

# 第二步输出：nx 图文件夹
nx_dir = os.path.join(base_dir, "nx")

# 第三步输出：注释后的 graph 文件夹
annot_dir = os.path.join(base_dir, "nx_p")


# =====================
# 创建输出文件夹
# =====================
os.makedirs(rna_info_dir, exist_ok=True)
os.makedirs(nx_dir, exist_ok=True)
os.makedirs(annot_dir, exist_ok=True)


# =====================
# Step 1
# bind_site_cif -> 1.p
# =====================
print("\n==============================")
print("[STEP 1] Extract RNA residues from CIF")
print("==============================")

RNA_ = process_all(
    pdb_path=bind_site_cif_dir,
    dump_path=rna_info_p
)

#print(f"[STEP 1 DONE] RNA residue dict saved to: {rna_info_p}")


# =====================
# Step 2
# 1.p + pickle/*.pickle -> nx/*.nx
# =====================
print("\n==============================")
print("[STEP 2] Build binding-site NX graphs")
print("==============================")

get_binding_site_graphs_all(
    RNA_full_residues=rna_info_p,
    RNA_full_path=bind_site_cif_dir,
    dump_path=nx_dir,
    pickle_dir=pickle_dir,
    non_binding=False
)

#print(f"[STEP 2 DONE] NX graphs saved to: {nx_dir}")


# =====================
# Step 3
# nx/*.nx -> nx_1/*_annot.p
# =====================
print("\n==============================")
print("[STEP 3] Annotate NX graphs")
print("==============================")

annotate_all(
    parallel=False,
    graph_path=nx_dir,
    dump_path=annot_dir,
    ablate="none"
)

#print(f"[STEP 3 DONE] Annotated graphs saved to: {annot_dir}")


print("\n==============================")
print("[ALL DONE]")
print("==============================")

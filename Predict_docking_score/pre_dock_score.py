import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np
from capsule_dock_score import FusionCapsuleRegression 


def load_model(model_path):

    model = FusionCapsuleRegression(
        RGCN_dim=16,
        EGNN_dim=128,
        SE_dim=1280,
        num_RGCN_caps=10,
        num_EGNN_caps=10,
        num_SE_caps=4,
        num_out_caps=6,
        capsule_dim=64,
        dropout_p=0.1,
        routing_iters=2,
        reg_hidden_dim=64
    )
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    return model


def test_model(model, X_Rmi_test, X_Con_test, X_SE_test, Y_test_label, Y_mean, Y_std):
    device = next(model.parameters()).device

    test_dataset = TensorDataset(X_Rmi_test, X_Con_test, X_SE_test, Y_test_label)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    test_pred_list = []



    with torch.no_grad():
        for vr, vc, vs, _ in tqdm(test_loader, desc="Testing"):
            vr, vc, vs = vr.to(device), vc.to(device), vs.to(device)

  
            pred, routing_c, u_hat = model(vr, vc, vs, return_coeff=True)

            test_pred_list.append(pred)


    test_pred = torch.cat(test_pred_list, dim=0).squeeze(-1)



    test_pred_denorm = test_pred * Y_std + Y_mean

    return test_pred_denorm


def build_inputs_from_con(con_csv,
                          rmi_csv,
                          se_csv,
                          label_csv,
                          verbose=True):


 
    df_con = pd.read_csv(con_csv, dtype=str)
    df_rmi = pd.read_csv(rmi_csv, dtype=str)
    df_se  = pd.read_csv(se_csv, dtype=str)
    df_label = pd.read_csv(label_csv, dtype=str)

    for df in (df_con, df_rmi, df_se, df_label):
        df.columns = [c.strip() for c in df.columns]


    def emb_cols(df):
        return [c for c in df.columns if c.startswith("emb_")]

    con_emb_cols = emb_cols(df_con)
    rmi_emb_cols = emb_cols(df_rmi)
    se_emb_cols  = emb_cols(df_se)

    if len(con_emb_cols) == 0:
        raise ValueError("Con 文件中找不到任何 emb_ 开头的列")
    if len(rmi_emb_cols) == 0:
        raise ValueError("RMI 文件中找不到任何 emb_ 开头的列")
    if len(se_emb_cols) == 0:
        raise ValueError("SE 文件中找不到任何 emb_ 开头的列")
    if 'pdb_id' not in df_con.columns:
        raise ValueError("Con 文件必须包含 pdb_id 列")
    if 'pdb_id' not in df_rmi.columns or 'pdb_id' not in df_se.columns or 'pdb_id' not in df_label.columns:
        raise ValueError("所有文件都必须包含 pdb_id 列（train_label 也需包含 pdb_id 与 RMSD）")
    if 'RMSD' not in df_label.columns:
        raise ValueError("label 文件中必须包含 RMSD 列")


    #df_rmi['prefix'] = df_rmi['pdb_id'].str[:4].str.upper()
    df_rmi['pdb_id_clean'] = (
    df_rmi['pdb_id']
    .astype(str)
    .str.strip()
    .str.replace(r'\.nx_annot\.p$', '', regex=True))
    df_se['prefix']  = df_se['pdb_id'].str[:4].str.upper()
    df_label['pdb_id'] = df_label['pdb_id'].str.strip()
    df_label_indexed = df_label.set_index('pdb_id')  
    #print (df_rmi['pdb_id'])

    rmi_map = {}
    for _, row in df_rmi.iterrows():
        k = row['pdb_id_clean']
        if k not in rmi_map:
           rmi_map[k] = row
           
    se_map = {}
    for _, row in df_se.iterrows():
        p = row['prefix']
        if p not in se_map:
            se_map[p] = row


    X_con_list = []
    X_rmi_list = []
    X_se_list = []
    Y_list = []
    matched_rows = []

    skipped_no_rmi = []
    skipped_no_se = []
    skipped_no_label = []

    for idx, crow in df_con.iterrows():
        con_pdb_full = str(crow['pdb_id']).strip()        
        prefix = con_pdb_full[:4].upper()              

    
        rmi_row = rmi_map.get(con_pdb_full, None)
        if rmi_row is None:
            skipped_no_rmi.append(con_pdb_full)
            if verbose:
                print(f"跳过（无 RMI）:  {con_pdb_full}")
            continue

       
        se_row = se_map.get(prefix, None)
        if se_row is None:
            skipped_no_se.append(con_pdb_full)
            if verbose:
                print(f"跳过（无 SE）: {con_pdb_full} (prefix={prefix})")
            continue

     
        if con_pdb_full not in df_label_indexed.index:
            skipped_no_label.append(con_pdb_full)
            if verbose:
                print(f"跳过（无 label）: {con_pdb_full}")
            continue

    
        try:
            con_vals = crow[con_emb_cols].astype(float).to_numpy(dtype=np.float32)
            rmi_vals = rmi_row[rmi_emb_cols].astype(float).to_numpy(dtype=np.float32)
            se_vals  = se_row[se_emb_cols].astype(float).to_numpy(dtype=np.float32)
            rmsd_val = float(df_label_indexed.loc[con_pdb_full]['RMSD'])
        except Exception as e:
            if verbose:
                print(f"数据转换错误，跳过 {con_pdb_full}: {e}")
            continue

        X_con_list.append(con_vals)
        X_rmi_list.append(rmi_vals)
        X_se_list.append(se_vals)
        Y_list.append([rmsd_val])

 
        matched_rows.append({
            'pdb_id': con_pdb_full,
            'prefix': prefix,
            **{f'con_{c}': crow[c] for c in con_emb_cols},
            **{f'rmi_{c}': rmi_row[c] for c in rmi_emb_cols},
            **{f'se_{c}': se_row[c] for c in se_emb_cols},
            'RMSD': rmsd_val
        })


    if len(X_con_list) == 0:
        raise RuntimeError("没有任何样本被匹配成功，请检查文件和匹配逻辑")

    X_Con_np = np.vstack(X_con_list)    # (N, con_dim)
    X_Rmi_np = np.vstack(X_rmi_list)    # (N, rmi_dim)
    X_Se_np  = np.vstack(X_se_list)     # (N, se_dim)
    Y_np     = np.array(Y_list, dtype=np.float32)  # (N,1)


    X_Con_train = torch.from_numpy(X_Con_np)
    X_Rmi_train = torch.from_numpy(X_Rmi_np)
    X_SE_train  = torch.from_numpy(X_Se_np)
    Y_train_label = torch.from_numpy(Y_np)

    matched_df = pd.DataFrame(matched_rows)

    if verbose:
        print(f"总样本数（匹配成功）: {len(X_Con_train)}")
        print(f"跳过（无 RMI）: {len(skipped_no_rmi)} , 无 SE: {len(skipped_no_se)}, 无 label: {len(skipped_no_label)}")

    return X_Rmi_train, X_Con_train, X_SE_train, Y_train_label, matched_df


##Provide your actual path.####
def main():

    X_Rmi_test, X_Con_test, X_SE_test, Y_test_label, matched_df_1 = build_inputs_from_con(
        con_csv='/ZHMolRSI/example/6HAG/interact_emb/model_interact_emb_out/6HAG_inter_emb.csv',
        rmi_csv='/ZHMolRSI/example/6HAG/pocket_emb/model_pocket_emb_out/6HAG_poc_emb.csv',
        se_csv='/ZHMolRSI/example/6HAG/sequence_emb/6HAG_se_emb.csv',
        label_csv='/ZHMolRSI/example/6HAG/label/6HAG_label.csv',
        verbose=True
    )


    Y_mean = 3.6449
    Y_std = 2.4720


    model = load_model('./model/capsule_best_d.pth')


    test_predictions = test_model(model, X_Rmi_test, X_Con_test, X_SE_test, Y_test_label, Y_mean, Y_std)
   
    results_df = pd.DataFrame({
        'pdb_id': matched_df_1['pdb_id'],
        'y_true': Y_test_label.squeeze(),
        'y_pred': test_predictions
          # 真实的 RMSD 标签
    })
   
    


    results_df2 = pd.DataFrame({
    'pdb_id': matched_df_1['pdb_id'],
    'y_true': Y_test_label.squeeze(),
    'y_pred': test_predictions,

})
    results_df.to_csv('/ZHMolRSI/example/6HAG/prediction/6HAG_dock_score_pre.csv', index=False, mode='w', header=True)

    
if __name__ == "__main__":
    main()


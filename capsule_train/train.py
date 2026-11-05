# -*- coding: utf-8 -*-

import os
import copy
import random
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from capsule import FusionCapsuleRegression




EPOCHS = 500
PATIENCE = 80
MIN_DELTA = 0.0         
LEARNING_RATE = 1e-4


TRAIN_POC = "./data/train_poc.csv"        
TRAIN_INTER = "./data/train_inter.csv"   
TRAIN_SE = "./data/train_SE.csv"          
TRAIN_LABEL = "./data/train_label.csv"    


VAL_POC = "./data/val_poc.csv"
VAL_INTER = "./data/val_inter.csv"
VAL_SE = "./data/val_SE.csv"             
VAL_LABEL = "./data/val_tabel.csv"     


OUT_DIR = "./results"
MODEL_DIR = os.path.join(OUT_DIR, "models")
VAL_PRED_DIR = os.path.join(OUT_DIR, "val_predictions")







def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    os.environ["PYTHONHASHSEED"] = str(seed)




def clean_pdb_id(value):
    s = str(value).strip().lower()

    if s.endswith(".nx_annot.p"):
        s = s[:-len(".nx_annot.p")]

    if s.endswith(".csv"):
        s = s[:-4]

    return s


def load_feature_file(path, expected_dim, sep=","):
    df = pd.read_csv(path, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]

    if "pdb_id" not in df.columns:
        raise ValueError(f"{path} 中没有 pdb_id 列")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    if len(emb_cols) != expected_dim:
        raise ValueError(
            f"{path} 特征维度错误：实际 {len(emb_cols)}，预期 {expected_dim}"
        )

    df["_pdb_clean"] = df["pdb_id"].map(clean_pdb_id)

    if df["_pdb_clean"].duplicated().any():
        dup = df.loc[
            df["_pdb_clean"].duplicated(keep=False),
            "_pdb_clean"
        ].tolist()

        raise ValueError(
            f"{path} 中存在重复 pdb_id，例如: {dup[:10]}"
        )

    for c in emb_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    if df[emb_cols].isna().any().any():
        bad_rows = df.index[
            df[emb_cols].isna().any(axis=1)
        ].tolist()

        raise ValueError(
            f"{path} 中存在非数值或 NaN 特征，"
            f"行号例如: {bad_rows[:10]}"
        )

    return df, emb_cols




def load_label_file(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "pdb_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{path} 必须包含 pdb_id 和 label 两列"
        )

    df["_pdb_clean"] = df["pdb_id"].map(clean_pdb_id)

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    if df["_pdb_clean"].duplicated().any():
        dup = df.loc[
            df["_pdb_clean"].duplicated(keep=False),
            "_pdb_clean"
        ].tolist()

        raise ValueError(
            f"{path} 中存在重复 pdb_id，例如: {dup[:10]}"
        )

    if df["label"].isna().any():
        bad_rows = df.index[
            df["label"].isna()
        ].tolist()

        raise ValueError(
            f"{path} 中存在无效 label，"
            f"行号例如: {bad_rows[:10]}"
        )

    return df



def check_same_ids(
    poc_ids,
    inter_ids,
    se_ids,
    label_ids,
    split_name
):
    all_ids = poc_ids | inter_ids | se_ids | label_ids

    errors = []

    for name, ids in [
        ("POC", poc_ids),
        ("INTER", inter_ids),
        ("SE", se_ids),
        ("LABEL", label_ids),
    ]:
        missing = sorted(all_ids - ids)

        if missing:
            errors.append(
                f"{name} 缺少 {len(missing)} 个，例如 {missing[:10]}"
            )

    if errors:
        raise RuntimeError(
            f"{split_name}: 四个文件的 pdb_id 不完全一致：\n"
            + "\n".join(errors)
        )




def build_dataset(
    poc_file,
    inter_file,
    se_file,
    label_file,
    split_name
):
    poc_df, poc_cols = load_feature_file(
        poc_file,
        expected_dim=16,
        sep=","
    )

    inter_df, inter_cols = load_feature_file(
        inter_file,
        expected_dim=128,
        sep=","
    )

    se_df, se_cols = load_feature_file(
        se_file,
        expected_dim=1280,
        sep="\t"
    )

    label_df = load_label_file(
        label_file
    )

    poc_ids = set(poc_df["_pdb_clean"])
    inter_ids = set(inter_df["_pdb_clean"])
    se_ids = set(se_df["_pdb_clean"])
    label_ids = set(label_df["_pdb_clean"])

    check_same_ids(
        poc_ids,
        inter_ids,
        se_ids,
        label_ids,
        split_name
    )

    common_ids = sorted(label_ids)



    poc_idx = poc_df.set_index("_pdb_clean")
    inter_idx = inter_df.set_index("_pdb_clean")
    se_idx = se_df.set_index("_pdb_clean")
    label_idx = label_df.set_index("_pdb_clean")

    X_poc = poc_idx.loc[
        common_ids,
        poc_cols
    ].to_numpy(
        dtype=np.float32
    )

    X_inter = inter_idx.loc[
        common_ids,
        inter_cols
    ].to_numpy(
        dtype=np.float32
    )

    X_se = se_idx.loc[
        common_ids,
        se_cols
    ].to_numpy(
        dtype=np.float32
    )

    y = label_idx.loc[
        common_ids,
        "label"
    ].to_numpy(
        dtype=np.float32
    )

    pdb_ids = np.array(
        common_ids
    )






    return (
        torch.from_numpy(X_poc),
        torch.from_numpy(X_inter),
        torch.from_numpy(X_se),
        torch.from_numpy(y),
        pdb_ids,
    )


# ============================================================
# Routing contribution
# ============================================================

def calculate_routing_contribution(
    model,
    routing_c,
    u_hat
):
    nr = model.num_RGCN_caps
    ne = model.num_GNN_caps
    ns = model.num_SE_caps

    energy = torch.norm(
        routing_c.unsqueeze(-1) * u_hat,
        dim=-1
    )

    sample_total_energy = (
        energy.sum(
            dim=(1, 2),
            keepdim=True
        )
        + 1e-8
    )

    energy_norm = (
        energy / sample_total_energy
    )

    rgcn_energy = (
        energy_norm[
            :,
            :nr,
            :
        ]
        .sum(dim=(1, 2))
        .mean()
    )

    gnn_energy = (
        energy_norm[
            :,
            nr:nr + ne,
            :
        ]
        .sum(dim=(1, 2))
        .mean()
    )

    se_energy = (
        energy_norm[
            :,
            nr + ne:nr + ne + ns,
            :
        ]
        .sum(dim=(1, 2))
        .mean()
    )

    total_energy = (
        rgcn_energy
        + gnn_energy
        + se_energy
        + 1e-8
    )

    return {
        "routing_RGCN":
            (rgcn_energy / total_energy).item(),

        "routing_GNN":
            (gnn_energy / total_energy).item(),

        "routing_SE":
            (se_energy / total_energy).item(),
    }

_RUN_CONFIG = (
    (1490, "capsule_seed1"),
    (1919, "capsule_seed2"),
    (6939, "capsule_best"),
    (12565, "capsule_seed3"),
    (14250, "capsule_seed4"),
)


def train_model(
    model,
    optimizer,
    criterion,
    X_poc_train,
    X_inter_train,
    X_se_train,
    y_train,
    X_poc_val,
    X_inter_val,
    X_se_val,
    y_val,
    epochs=500,
    patience=80,
    min_delta=0.0,
):
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    best_routing = {
        "routing_RGCN": 0.0,
        "routing_GNN": 0.0,
        "routing_SE": 0.0,
    }

    for epoch in range(
        1,
        epochs + 1
    ):

        # ====================================================
        # Train
        # ====================================================

        model.train()

        pred_train, _, _ = model(
            X_poc_train,
            X_inter_train,
            X_se_train,
            return_coeff=True,
        )

        train_loss = criterion(
            pred_train,
            y_train.view(-1)
        )

        if not torch.isfinite(
            train_loss
        ):

            raise RuntimeError(
                f"epoch {epoch}: "
                f"train_loss 出现 NaN/Inf"
            )

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()


        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        with torch.no_grad():

            pred_val, routing_c, u_hat = model(
                X_poc_val,
                X_inter_val,
                X_se_val,
                return_coeff=True,
            )

            val_loss = criterion(
                pred_val,
                y_val.view(-1)
            ).item()

            if not np.isfinite(
                val_loss
            ):

                raise RuntimeError(
                    f"epoch {epoch}: "
                    f"val_loss 出现 NaN/Inf"
                )

            routing = (
                calculate_routing_contribution(
                    model,
                    routing_c,
                    u_hat
                )
            )


        train_losses.append(
            train_loss.item()
        )

        val_losses.append(
            val_loss
        )


        # ====================================================
        # Early stopping
        # ====================================================

        if (
            val_loss
            < best_val_loss - min_delta
        ):

            best_val_loss = val_loss

            best_epoch = epoch

            epochs_no_improve = 0

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            best_routing = routing.copy()

        else:

            epochs_no_improve += 1


        if (
            epoch == 1
            or epoch % 20 == 0
            or epochs_no_improve == 0
        ):

            gate_np = (
                torch.softmax(
                    model.modality_gate,
                    dim=0
                )
                .detach()
                .cpu()
                .numpy()
            )

            print(
                f"Epoch {epoch:4d} | "
                f"train={train_loss.item():.6f} | "
                f"val={val_loss:.6f} | "
                f"best={best_val_loss:.6f} | "
                f"no_improve={epochs_no_improve:3d} | "
                f"gate="
                f"[{gate_np[0]:.3f}, "
                f"{gate_np[1]:.3f}, "
                f"{gate_np[2]:.3f}]"
            )


        if (
            epochs_no_improve
            >= patience
        ):

            print(
                f"Early stopping: "
                f"epoch={epoch}, "
                f"best_epoch={best_epoch}"
            )

            break


    if best_model_state is None:

        raise RuntimeError(
            "训练过程中没有得到有效的最佳模型"
        )



    model.load_state_dict(
        best_model_state
    )


    return (
        model,
        best_epoch,
        best_val_loss,
        best_routing,
        train_losses,
        val_losses,
    )


def evaluate_validation(
    model,
    X_poc_val,
    X_inter_val,
    X_se_val,
    y_val
):
    model.eval()

    with torch.no_grad():

        val_pred, _, _ = model(
            X_poc_val,
            X_inter_val,
            X_se_val,
            return_coeff=True,
        )


    y_true = (
        y_val
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    y_pred = (
        val_pred
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = sqrt(
        mse
    )

    mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )



    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):

        pearson = np.nan
        spearman = np.nan

    else:

        pearson = np.corrcoef(
            y_true,
            y_pred
        )[0, 1]

        spearman, _ = spearmanr(
            y_true,
            y_pred
        )


    gate_np = (
        torch.softmax(
            model.modality_gate,
            dim=0
        )
        .detach()
        .cpu()
        .numpy()
    )


    return (
        y_true,
        y_pred,
        rmse,
        mae,
        r2,
        pearson,
        spearman,
        gate_np,
    )




def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        VAL_PRED_DIR,
        exist_ok=True
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )




    (
        X_poc_train,
        X_inter_train,
        X_se_train,
        y_train,
        train_pdb_ids,
    ) = build_dataset(
        TRAIN_POC,
        TRAIN_INTER,
        TRAIN_SE,
        TRAIN_LABEL,
        "TRAIN",
    )


    (
        X_poc_val,
        X_inter_val,
        X_se_val,
        y_val,
        val_pdb_ids,
    ) = build_dataset(
        VAL_POC,
        VAL_INTER,
        VAL_SE,
        VAL_LABEL,
        "VALIDATION",
    )








    X_poc_train = X_poc_train.to(
        device
    )

    X_inter_train = X_inter_train.to(
        device
    )

    X_se_train = X_se_train.to(
        device
    )

    y_train = y_train.to(
        device
    )


    X_poc_val = X_poc_val.to(
        device
    )

    X_inter_val = X_inter_val.to(
        device
    )

    X_se_val = X_se_val.to(
        device
    )

    y_val = y_val.to(
        device
    )


    all_results = []

    for run_index, (seed, model_name) in enumerate(_RUN_CONFIG, start=1):
        model_file = os.path.join(MODEL_DIR, f"{model_name}.pth")
        pred_file = os.path.join(VAL_PRED_DIR, f"{model_name}_val_predictions.csv")
        metrics_file = os.path.join(VAL_PRED_DIR, f"{model_name}_val_metrics.csv")
        print(f"\n{'=' * 60}\n{model_name} ({run_index}/{len(_RUN_CONFIG)})\n{'=' * 60}")

        row = {
            "model_name": model_name,
            "model_file": model_file,
            "val_predictions_file": pred_file,
            "val_metrics_file": metrics_file,
        }
        try:

            set_seed(seed)
            model = FusionCapsuleRegression(
                RGCN_dim=16, GNN_dim=128, SE_dim=1280,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.MSELoss()

            (
                model, best_epoch, best_val_loss, best_routing,
                train_losses, val_losses,
            ) = train_model(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                X_poc_train=X_poc_train,
                X_inter_train=X_inter_train,
                X_se_train=X_se_train,
                y_train=y_train,
                X_poc_val=X_poc_val,
                X_inter_val=X_inter_val,
                X_se_val=X_se_val,
                y_val=y_val,
                epochs=EPOCHS,
                patience=PATIENCE,
                min_delta=MIN_DELTA,
            )

            (
                y_true, y_pred, val_rmse, val_mae, val_r2,
                val_pearson, val_spearman, gate_np,
            ) = evaluate_validation(
                model, X_poc_val, X_inter_val, X_se_val, y_val,
            )


            torch.save(model.state_dict(), model_file)
            pd.DataFrame({
                "pdb_id": val_pdb_ids,
                "y_true": y_true,
                "y_pred": y_pred,
            }).to_csv(pred_file, index=False)

            row.update({
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "gate_poc_rgcn": gate_np[0],
                "gate_inter_gnn": gate_np[1],
                "gate_se": gate_np[2],
                "routing_poc_rgcn": best_routing["routing_RGCN"],
                "routing_inter_gnn": best_routing["routing_GNN"],
                "routing_se": best_routing["routing_SE"],
                "status": "OK",
            })
            pd.DataFrame([row]).to_csv(metrics_file, index=False)
            print(
                f"[{model_name}] best_epoch={best_epoch}, "
                f"val_loss={best_val_loss:.6f}, RMSE={val_rmse:.6f}, "
                f"MAE={val_mae:.6f}, R2={val_r2:.6f}, "
                f"Pearson={val_pearson:.6f}, Spearman={val_spearman:.6f}"
            )
        except Exception as exc:
            row["status"] = f"ERROR: {exc}"
            print(f"[ERROR] {model_name}: {exc}")

        all_results.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(all_results)
    ok_df = result_df[result_df["status"] == "OK"].copy()
    if ok_df.empty:
        raise RuntimeError("所有配置均训练失败，详情见终端错误信息。")

    print(f"\n训练结束：成功 {len(ok_df)}/{len(_RUN_CONFIG)}")
    print(f"模型目录：{MODEL_DIR}")
    print(f"验证结果目录：{VAL_PRED_DIR}")


if __name__ == "__main__":
    main()

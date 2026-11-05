# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from capsule_affi import FusionCapsuleRegression


# ============================================================
# 参数
# ============================================================

# 训练阶段使用的 Min-Max 参数
Y_MIN = 2.508638306
Y_MAX = 10.95860731

BATCH_SIZE = 128

# 包含所有 .pth 模型的文件夹
MODEL_DIR = "./model"

# 是否递归搜索 model 下所有子文件夹中的 .pth
RECURSIVE_MODEL_SEARCH = True

# 新测试集
TEST_POC = "./xdock_pose/RSI_a12_x_poc.csv"          # 16维；例如 pdb_1.nx_annot.p
TEST_INTER = "./xdock_pose/RSI_a12_x_inter.csv"      # 128维；例如 pdb_1
TEST_SE = "./xdock_pose/RSI_a12_x_se.csv"            # 1280维；例如 pdb_1.csv
TEST_LABEL = "./xdock_pose/RSI_a12_x_label.csv"      # pdb_id,RMSD

# 每个模型的预测结果保存目录
OUTPUT_DIR = "./pre_xdock_result"

# 所有模型预测指标汇总
METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "all_result_metrics.csv"
)


# ============================================================
# 统一 pdb_id
# ============================================================

def clean_pdb_id(value):
    """
    统一完整 pose ID。

    例如：
        pdb_1.nx_annot.p -> pdb_1
        pdb_1            -> pdb_1
        pdb_1.csv        -> pdb_1
    """
    s = str(value).strip().lower()

    if s.endswith(".nx_annot.p"):
        s = s[:-len(".nx_annot.p")]

    if s.endswith(".csv"):
        s = s[:-4]

    return s


# ============================================================
# 获取 emb_* 特征列
# ============================================================

def get_emb_cols(
    df,
    expected_dim,
    file_name
):
    emb_cols = [
        c for c in df.columns
        if str(c).startswith("emb_")
    ]

    if len(emb_cols) != expected_dim:
        raise ValueError(
            f"{file_name} 特征维度错误："
            f"实际 {len(emb_cols)}，预期 {expected_dim}"
        )

    return emb_cols


# ============================================================
# 数值检查
# ============================================================

def convert_feature_to_numeric(
    df,
    emb_cols,
    file_name
):
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
            f"{file_name} 中存在 NaN 或非数值特征，"
            f"行号例如: {bad_rows[:10]}"
        )


# ============================================================
# 构建测试集
#
# 以 INTER 的顺序为最终顺序：
#
# INTER : 完整 pose ID
# POC   : 去掉 .nx_annot.p 后精确匹配
# LABEL : 完整 pose ID 精确匹配
# SE    : 去掉 .csv 后按完整 pdb_id 精确匹配
# ============================================================

def build_test_dataset(
    inter_file,
    poc_file,
    se_file,
    label_file
):

    # --------------------------------------------------------
    # 读取文件
    # --------------------------------------------------------

    df_inter = pd.read_csv(
        inter_file
    )

    df_poc = pd.read_csv(
        poc_file
    )

    # 当前 RSI_a12_x_se.csv 是逗号分隔
    df_se = pd.read_csv(
        se_file
    )

    df_label = pd.read_csv(
        label_file
    )


    # --------------------------------------------------------
    # 清理列名
    # --------------------------------------------------------

    for df in (
        df_inter,
        df_poc,
        df_se,
        df_label
    ):

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]


    # --------------------------------------------------------
    # 检查 pdb_id
    # --------------------------------------------------------

    for name, df in [
        ("INTER", df_inter),
        ("POC", df_poc),
        ("SE", df_se),
        ("LABEL", df_label),
    ]:

        if "pdb_id" not in df.columns:

            raise ValueError(
                f"{name} 文件没有 pdb_id 列"
            )


    # --------------------------------------------------------
    # 标签必须包含 RMSD
    # --------------------------------------------------------

    if "RMSD" not in df_label.columns:

        raise ValueError(
            "RSI_a12_x_label.csv 必须包含 RMSD 列"
        )


    # --------------------------------------------------------
    # 检查特征维度
    # --------------------------------------------------------

    inter_cols = get_emb_cols(
        df_inter,
        expected_dim=128,
        file_name="RSI_a12_x_inter.csv"
    )

    poc_cols = get_emb_cols(
        df_poc,
        expected_dim=16,
        file_name="RSI_a12_x_poc.csv"
    )

    se_cols = get_emb_cols(
        df_se,
        expected_dim=1280,
        file_name="RSI_a12_x_se.csv"
    )


    # --------------------------------------------------------
    # 转数值并检查 NaN
    # --------------------------------------------------------

    convert_feature_to_numeric(
        df_inter,
        inter_cols,
        "RSI_a12_x_inter.csv"
    )

    convert_feature_to_numeric(
        df_poc,
        poc_cols,
        "RSI_a12_x_poc.csv"
    )

    convert_feature_to_numeric(
        df_se,
        se_cols,
        "RSI_a12_x_se.csv"
    )

    df_label["RMSD"] = pd.to_numeric(
        df_label["RMSD"],
        errors="coerce"
    )

    if df_label["RMSD"].isna().any():

        bad_rows = df_label.index[
            df_label["RMSD"].isna()
        ].tolist()

        raise ValueError(
            f"RSI_a12_x_label.csv 中存在无效 RMSD，"
            f"行号例如: {bad_rows[:10]}"
        )


    # --------------------------------------------------------
    # 清洗 pdb_id
    # --------------------------------------------------------

    for name, df in [("INTER", df_inter), ("POC", df_poc),
                     ("SE", df_se), ("LABEL", df_label)]:
        if df.empty:
            raise ValueError(f"{name} 文件没有数据行")
        if df["pdb_id"].isna().any() or df["pdb_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} 文件包含空 pdb_id")

    df_inter["_pdb_clean"] = (
        df_inter["pdb_id"].map(
            clean_pdb_id
        )
    )

    df_poc["_pdb_clean"] = (
        df_poc["pdb_id"].map(
            clean_pdb_id
        )
    )

    df_label["_pdb_clean"] = (
        df_label["pdb_id"].map(
            clean_pdb_id
        )
    )

    df_se["_pdb_clean"] = (
        df_se["pdb_id"].map(
            clean_pdb_id
        )
    )


    # --------------------------------------------------------
    # 检查重复
    # --------------------------------------------------------

    for name, df in [
        ("INTER", df_inter),
        ("POC", df_poc),
        ("LABEL", df_label),
        ("SE", df_se),
    ]:

        if df["_pdb_clean"].duplicated().any():

            duplicated = df.loc[
                df["_pdb_clean"].duplicated(
                    keep=False
                ),
                "_pdb_clean"
            ].tolist()

            raise ValueError(
                f"{name} 中存在重复 pdb_id，"
                f"例如: {duplicated[:10]}"
            )


    # --------------------------------------------------------
    # INTER 顺序作为测试顺序
    # --------------------------------------------------------

    test_ids = (
        df_inter["_pdb_clean"]
        .tolist()
    )

    # SE 与其他输入使用相同的完整 ID，不截取前四位。
    test_se_ids = list(test_ids)


    # --------------------------------------------------------
    # 建立索引
    # --------------------------------------------------------

    inter_idx = df_inter.set_index(
        "_pdb_clean"
    )

    poc_idx = df_poc.set_index(
        "_pdb_clean"
    )

    label_idx = df_label.set_index(
        "_pdb_clean"
    )

    se_idx = df_se.set_index(
        "_pdb_clean"
    )


    # --------------------------------------------------------
    # 严格检查 pose 级匹配
    # --------------------------------------------------------

    inter_set = set(
        test_ids
    )

    poc_set = set(
        poc_idx.index
    )

    label_set = set(
        label_idx.index
    )


    missing_poc = sorted(
        inter_set - poc_set
    )

    missing_label = sorted(
        inter_set - label_set
    )


    if missing_poc:

        raise RuntimeError(
            f"POC 缺少 {len(missing_poc)} 个 pose，"
            f"例如: {missing_poc[:10]}"
        )


    if missing_label:

        raise RuntimeError(
            f"LABEL 缺少 {len(missing_label)} 个 pose，"
            f"例如: {missing_label[:10]}"
        )


    # --------------------------------------------------------
    # 检查 SE 匹配
    # --------------------------------------------------------

    se_set = set(
        se_idx.index
    )

    missing_se = sorted(
        set(test_se_ids) - se_set
    )


    if missing_se:

        raise RuntimeError(
            f"SE 缺少 {len(missing_se)} 个 PDB，"
            f"例如: {missing_se[:10]}"
        )


    # --------------------------------------------------------
    # 构建矩阵
    # --------------------------------------------------------

    X_inter = inter_idx.loc[
        test_ids,
        inter_cols
    ].to_numpy(
        dtype=np.float32
    )

    X_poc = poc_idx.loc[
        test_ids,
        poc_cols
    ].to_numpy(
        dtype=np.float32
    )

    X_se = se_idx.loc[
        test_se_ids,
        se_cols
    ].to_numpy(
        dtype=np.float32
    )

    y_rmsd = label_idx.loc[
        test_ids,
        "RMSD"
    ].to_numpy(
        dtype=np.float32
    )


    # --------------------------------------------------------
    # 输出检查
    # --------------------------------------------------------

    print("=" * 80)
    print("测试集匹配完成")
    print(f"Pose 数量   : {len(test_ids)}")
    print(f"PDB 数量    : {len(set(test_se_ids))}")
    print(f"POC shape   : {X_poc.shape}")
    print(f"INTER shape : {X_inter.shape}")
    print(f"SE shape    : {X_se.shape}")
    print(f"RMSD shape  : {y_rmsd.shape}")
    print(
        f"RMSD range  : "
        f"{y_rmsd.min():.6f} ~ "
        f"{y_rmsd.max():.6f}"
    )
    print("=" * 80)


    return (
        torch.from_numpy(X_poc),
        torch.from_numpy(X_inter),
        torch.from_numpy(X_se),
        torch.from_numpy(y_rmsd),
        np.array(test_ids)
    )


# ============================================================
# 创建模型结构
# ============================================================

def create_model(
    device
):

    model = FusionCapsuleRegression(
        RGCN_dim=16,
        GNN_dim=128,
        SE_dim=1280,
        num_RGCN_caps=4,
        num_GNN_caps=4,
        num_SE_caps=2,
        num_out_caps=4,
        capsule_dim=32,
        dropout_p=0.1
    ).to(
        device
    )

    return model


# ============================================================
# 加载一个 .pth
# ============================================================

def load_model(
    model_path,
    device
):

    model = create_model(
        device
    )


    try:

        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )

    except TypeError:

        state_dict = torch.load(
            model_path,
            map_location=device
        )


    # --------------------------------------------------------
    # 兼容以下保存格式：
    #
    # 1. torch.save(model.state_dict(), path)
    # 2. {"model_state_dict": ...}
    # 3. {"state_dict": ...}
    # --------------------------------------------------------

    if isinstance(
        state_dict,
        dict
    ):

        if (
            "model_state_dict"
            in state_dict
        ):

            state_dict = state_dict[
                "model_state_dict"
            ]

        elif (
            "state_dict"
            in state_dict
        ):

            state_dict = state_dict[
                "state_dict"
            ]


    # --------------------------------------------------------
    # 如果 DataParallel 保存成 module.xxx
    # 自动去掉 module.
    # --------------------------------------------------------

    if (
        isinstance(state_dict, dict)
        and len(state_dict) > 0
        and all(
            str(k).startswith("module.")
            for k in state_dict.keys()
        )
    ):

        state_dict = {
            str(k)[len("module."):]: v
            for k, v in state_dict.items()
        }


    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# 创建 DataLoader
#
# 测试集只构建一次，所有模型复用同一个 DataLoader
# ============================================================

def create_test_loader(
    X_poc,
    X_inter,
    X_se,
    batch_size
):

    dataset = TensorDataset(
        X_poc,
        X_inter,
        X_se
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader


# ============================================================
# 预测一个模型
# ============================================================

def predict_one_model(
    model,
    dataloader,
    device
):

    pred_norm_list = []


    with torch.no_grad():

        for (
            batch_poc,
            batch_inter,
            batch_se
        ) in dataloader:

            batch_poc = batch_poc.to(
                device
            )

            batch_inter = batch_inter.to(
                device
            )

            batch_se = batch_se.to(
                device
            )


            pred_norm, _, _ = model(
                batch_poc,
                batch_inter,
                batch_se,
                return_coeff=True
            )


            pred_norm_list.append(
                pred_norm.detach().cpu()
            )


    pred_norm = torch.cat(
        pred_norm_list,
        dim=0
    ).numpy().reshape(-1)


    # --------------------------------------------------------
    # 反归一化
    #
    # y = y_norm * (Y_MAX - Y_MIN) + Y_MIN
    # --------------------------------------------------------

    pred_denorm = (
        pred_norm
        * (Y_MAX - Y_MIN)
        + Y_MIN
    )


    return (
        pred_norm,
        pred_denorm
    )


# ============================================================
# 找到 model 文件夹下所有 .pth
# ============================================================

def find_model_files(
    model_dir,
    recursive=True
):

    model_files = []


    if recursive:

        for root, dirs, files in os.walk(
            model_dir
        ):

            dirs.sort()

            for filename in sorted(
                files
            ):

                if filename.lower().endswith(
                    ".pth"
                ):

                    model_files.append(
                        os.path.join(
                            root,
                            filename
                        )
                    )

    else:

        for filename in sorted(
            os.listdir(
                model_dir
            )
        ):

            path = os.path.join(
                model_dir,
                filename
            )

            if (
                os.path.isfile(path)
                and filename.lower().endswith(
                    ".pth"
                )
            ):

                model_files.append(
                    path
                )


    return sorted(
        model_files
    )


# ============================================================
# 根据模型相对路径生成安全的输出文件名
#
# model/seed1.pth
# -> seed1.csv
#
# model/a/seed1.pth
# -> a__seed1.csv
# ============================================================

def model_to_output_name(
    model_path
):

    relative_path = os.path.relpath(
        model_path,
        MODEL_DIR
    )

    relative_no_ext = os.path.splitext(
        relative_path
    )[0]

    safe_name = re.sub(
        r"[\\/]+",
        "__",
        relative_no_ext
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        safe_name
    )


    return (
        safe_name
        + ".csv"
    )


# ============================================================
# 主程序
# ============================================================

def calculate_prediction_metrics(y_true, y_pred):
    """对保存到预测 CSV 的真实值和反归一化预测值计算四项指标。"""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size == 0 or true.shape != pred.shape:
        raise ValueError("真实值与预测值必须非空且长度一致")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("真实值或预测值包含 NaN/Inf")

    error = pred - true
    metrics = {
        "PCC": np.nan,
        "SPCC": np.nan,
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
    }
    # 少于两个样本或任一数组为常数时，相关系数未定义，CSV 中留空。
    if true.size >= 2 and np.ptp(true) > 0 and np.ptp(pred) > 0:
        metrics["PCC"] = float(np.corrcoef(true, pred)[0, 1])
        # 并列值采用平均秩。
        true_rank = pd.Series(true).rank(method="average").to_numpy()
        pred_rank = pd.Series(pred).rank(method="average").to_numpy()
        metrics["SPCC"] = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return metrics


def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )


    # --------------------------------------------------------
    # 检查 model 文件夹
    # --------------------------------------------------------

    if not os.path.isdir(
        MODEL_DIR
    ):

        raise FileNotFoundError(
            f"没有找到模型文件夹: {MODEL_DIR}"
        )


    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # 只构建一次测试集
    # --------------------------------------------------------

    (
        X_poc_test,
        X_inter_test,
        X_se_test,
        y_rmsd,
        pdb_ids
    ) = build_test_dataset(
        inter_file=TEST_INTER,
        poc_file=TEST_POC,
        se_file=TEST_SE,
        label_file=TEST_LABEL
    )


    y_true = (
        y_rmsd
        .cpu()
        .numpy()
        .reshape(-1)
    )


    # --------------------------------------------------------
    # DataLoader 也只构建一次
    # --------------------------------------------------------

    test_loader = create_test_loader(
        X_poc=X_poc_test,
        X_inter=X_inter_test,
        X_se=X_se_test,
        batch_size=BATCH_SIZE
    )


    # --------------------------------------------------------
    # 找所有模型
    # --------------------------------------------------------

    model_files = find_model_files(
        MODEL_DIR,
        recursive=RECURSIVE_MODEL_SEARCH
    )


    if len(model_files) == 0:

        raise RuntimeError(
            f"{MODEL_DIR} 中没有找到任何 .pth 文件"
        )


    print()
    print("=" * 80)
    print(
        f"共找到 {len(model_files)} 个 .pth 模型"
    )
    print("=" * 80)


    for i, model_path in enumerate(
        model_files,
        start=1
    ):

        print(
            f"{i:4d}. "
            f"{os.path.relpath(model_path, MODEL_DIR)}"
        )


    # --------------------------------------------------------
    # 逐个测试
    # --------------------------------------------------------

    summary_rows = []


    for model_index, model_path in enumerate(
        model_files,
        start=1
    ):

        relative_model = os.path.relpath(
            model_path,
            MODEL_DIR
        )

        output_name = model_to_output_name(
            model_path
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            output_name
        )


        print()
        print("=" * 80)

        print(
            f"[{model_index}/{len(model_files)}] "
            f"测试模型: {relative_model}"
        )

        print(
            f"输出文件: {output_file}"
        )


        try:

            # ------------------------------------------------
            # 加载模型
            # ------------------------------------------------

            model = load_model(
                model_path,
                device
            )


            # ------------------------------------------------
            # 预测
            # ------------------------------------------------

            (
                y_pred_norm,
                y_pred
            ) = predict_one_model(
                model=model,
                dataloader=test_loader,
                device=device
            )


            # ------------------------------------------------
            # 保存这个模型的所有 pose 预测
            # ------------------------------------------------

            result_df = pd.DataFrame({
                "pdb_id": pdb_ids,
                "y_true": y_true,
                "y_pred": y_pred
            })


            metrics = calculate_prediction_metrics(y_true, y_pred)

            result_df.to_csv(
                output_file,
                index=False
            )


            # ------------------------------------------------
            # 记录汇总
            # ------------------------------------------------

            summary_rows.append({
                "file_name": output_name,
                "n_samples": len(y_true),
                **metrics,
                "status": "OK",
            })


            print(
                f"[DONE] "
                f"prediction range = "
                f"{y_pred.min():.6f} ~ "
                f"{y_pred.max():.6f}"
            )


        except Exception as e:

            print(
                f"[ERROR] {relative_model}: {e}"
            )


            summary_rows.append({
                "file_name": output_name,
                "n_samples": len(y_true),
                "PCC": np.nan,
                "SPCC": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "status": f"ERROR: {e}",
            })


        finally:

            # ------------------------------------------------
            # 删除当前模型，释放显存
            # ------------------------------------------------

            if "model" in locals():

                del model


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


        # ----------------------------------------------------
        # 每测完一个模型就保存一次 summary
        # 防止长时间运行过程中断导致记录丢失
        # ----------------------------------------------------

        pd.DataFrame(
            summary_rows
        ).to_csv(
            METRICS_FILE,
            index=False
        )


    # --------------------------------------------------------
    # 最终统计
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    success_num = int(
        (
            summary_df["status"]
            == "OK"
        ).sum()
    )

    failed_num = (
        len(summary_df)
        - success_num
    )


    print()
    print("=" * 80)
    print("全部模型测试结束")
    print(
        f"模型总数    : {len(model_files)}"
    )
    print(
        f"成功        : {success_num}"
    )
    print(
        f"失败        : {failed_num}"
    )
    print(
        f"预测结果目录: {OUTPUT_DIR}"
    )
    print(
        f"汇总文件    : {METRICS_FILE}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

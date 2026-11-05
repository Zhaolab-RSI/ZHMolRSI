# -*- coding: utf-8 -*-

import os
import re
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from capsule_dock import FusionCapsuleRegression




MODEL_DIR = "./model"
OUTPUT_DIR = "./result/RSI_d3_x"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CON_CSV = "./RSI_d3/xdock_pose/31_inter.csv"
RMI_CSV = "./RSI_d3/xdock_pose/31_poc.csv"
SE_CSV = "./RSI_d3/xdock_pose/31_se.csv"
LABEL_CSV = "./RSI_d3/xdock_pose/31_label.csv"


TOP_K_LIST = [1, 2, 3, 4, 8, 20]
RMSD_THRESHOLD = 2.0
SUCCESS_XLSX = os.path.join(
    OUTPUT_DIR,
    os.path.splitext(os.path.basename(CON_CSV))[0] + "_success_summary.xlsx"
)


Y_MEAN = 3.6449
Y_STD = 2.4720

# GPU / CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




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

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model



def test_model(
    model,
    X_Rmi_test,
    X_Con_test,
    X_SE_test,
    Y_test_label,
    Y_mean,
    Y_std
):

    device = next(
        model.parameters()
    ).device

    test_dataset = TensorDataset(
        X_Rmi_test,
        X_Con_test,
        X_SE_test,
        Y_test_label
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    test_pred_list = []

    with torch.no_grad():

        for vr, vc, vs, _ in tqdm(
            test_loader,
            desc="Testing",
            leave=False
        ):

            vr = vr.to(device)
            vc = vc.to(device)
            vs = vs.to(device)

           
            pred, routing_c, u_hat = model(
                vr,
                vc,
                vs,
                return_coeff=True
            )

         
            test_pred_list.append(
                pred.detach().cpu()
            )


    test_pred = torch.cat(
        test_pred_list,
        dim=0
    ).squeeze(-1)




    test_pred_denorm = (
        test_pred * Y_std
        + Y_mean
    )

    return test_pred_denorm




def build_inputs_from_con(
    con_csv,
    rmi_csv,
    se_csv,
    label_csv,
    verbose=True
):

    df_con = pd.read_csv(
        con_csv,
        dtype=str
    )

    df_rmi = pd.read_csv(
        rmi_csv,
        dtype=str
    )

    df_se = pd.read_csv(
        se_csv,
        dtype=str
    )

    df_label = pd.read_csv(
        label_csv,
        dtype=str
    )




    for df in (
        df_con,
        df_rmi,
        df_se,
        df_label
    ):

        df.columns = [
            c.strip()
            for c in df.columns
        ]


    def emb_cols(df):

        return [
            c
            for c in df.columns
            if c.startswith("emb_")
        ]


    con_emb_cols = emb_cols(
        df_con
    )

    rmi_emb_cols = emb_cols(
        df_rmi
    )

    se_emb_cols = emb_cols(
        df_se
    )


    # ======================================================
    # 检查 embedding
    # ======================================================

    if len(con_emb_cols) == 0:

        raise ValueError(
            "Con 文件中找不到任何 emb_ 开头的列"
        )

    if len(rmi_emb_cols) == 0:

        raise ValueError(
            "RMI 文件中找不到任何 emb_ 开头的列"
        )

    if len(se_emb_cols) == 0:

        raise ValueError(
            "SE 文件中找不到任何 emb_ 开头的列"
        )


    # ======================================================
    # 检查 pdb_id
    # ======================================================

    if "pdb_id" not in df_con.columns:

        raise ValueError(
            "Con 文件必须包含 pdb_id 列"
        )

    if (
        "pdb_id" not in df_rmi.columns
        or
        "pdb_id" not in df_se.columns
        or
        "pdb_id" not in df_label.columns
    ):

        raise ValueError(
            "所有文件都必须包含 pdb_id 列"
        )

    if "RMSD" not in df_label.columns:

        raise ValueError(
            "label 文件中必须包含 RMSD 列"
        )


    # ======================================================
    # RMI pdb_id 清理
    #
    # xxx_lig_1.nx_annot.p
    # ->
    # xxx_lig_1
    # ======================================================

    df_rmi["pdb_id_clean"] = (
        df_rmi["pdb_id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.nx_annot\.p$",
            "",
            regex=True
        )
    )


    # ======================================================
    # SE 使用前4字符匹配
    # ======================================================

    df_se["prefix"] = (
        df_se["pdb_id"]
        .astype(str)
        .str[:4]
        .str.upper()
    )


    # ======================================================
    # label 清理
    # ======================================================

    df_label["pdb_id"] = (
        df_label["pdb_id"]
        .astype(str)
        .str.strip()
    )

    df_label_indexed = (
        df_label.set_index(
            "pdb_id"
        )
    )


    # ======================================================
    # RMI mapping
    # ======================================================

    rmi_map = {}

    for _, row in df_rmi.iterrows():

        k = row[
            "pdb_id_clean"
        ]

        if k not in rmi_map:

            rmi_map[k] = row


    # ======================================================
    # SE mapping
    # ======================================================

    se_map = {}

    for _, row in df_se.iterrows():

        p = row[
            "prefix"
        ]

        if p not in se_map:

            se_map[p] = row


    # ======================================================
    # 构建样本
    # ======================================================

    X_con_list = []
    X_rmi_list = []
    X_se_list = []
    Y_list = []

    matched_rows = []

    skipped_no_rmi = []
    skipped_no_se = []
    skipped_no_label = []


    for idx, crow in df_con.iterrows():

        con_pdb_full = str(
            crow["pdb_id"]
        ).strip()

        prefix = (
            con_pdb_full[:4]
            .upper()
        )


        # ==================================================
        # RMI 精确匹配
        # ==================================================

        rmi_row = rmi_map.get(
            con_pdb_full,
            None
        )

        if rmi_row is None:

            skipped_no_rmi.append(
                con_pdb_full
            )

            if verbose:

                print(
                    f"跳过（无 RMI）: "
                    f"{con_pdb_full}"
                )

            continue


        # ==================================================
        # SE 匹配
        # ==================================================

        se_row = se_map.get(
            prefix,
            None
        )

        if se_row is None:

            skipped_no_se.append(
                con_pdb_full
            )

            if verbose:

                print(
                    f"跳过（无 SE）: "
                    f"{con_pdb_full} "
                    f"(prefix={prefix})"
                )

            continue


        # ==================================================
        # label 精确匹配
        # ==================================================

        if (
            con_pdb_full
            not in
            df_label_indexed.index
        ):

            skipped_no_label.append(
                con_pdb_full
            )

            if verbose:

                print(
                    f"跳过（无 label）: "
                    f"{con_pdb_full}"
                )

            continue


        # ==================================================
        # 提取 embedding
        # ==================================================

        try:

            con_vals = (
                crow[
                    con_emb_cols
                ]
                .astype(float)
                .to_numpy(
                    dtype=np.float32
                )
            )

            rmi_vals = (
                rmi_row[
                    rmi_emb_cols
                ]
                .astype(float)
                .to_numpy(
                    dtype=np.float32
                )
            )

            se_vals = (
                se_row[
                    se_emb_cols
                ]
                .astype(float)
                .to_numpy(
                    dtype=np.float32
                )
            )

            rmsd_val = float(
                df_label_indexed.loc[
                    con_pdb_full
                ]["RMSD"]
            )


        except Exception as e:

            if verbose:

                print(
                    f"数据转换错误，"
                    f"跳过 {con_pdb_full}: "
                    f"{e}"
                )

            continue


        X_con_list.append(
            con_vals
        )

        X_rmi_list.append(
            rmi_vals
        )

        X_se_list.append(
            se_vals
        )

        Y_list.append([
            rmsd_val
        ])


        matched_rows.append({

            "pdb_id":
                con_pdb_full,

            "prefix":
                prefix,

            "RMSD":
                rmsd_val
        })


    # ======================================================
    # 检查是否匹配成功
    # ======================================================

    if len(X_con_list) == 0:

        raise RuntimeError(
            "没有任何样本被匹配成功，"
            "请检查文件和匹配逻辑"
        )


    # ======================================================
    # numpy
    # ======================================================

    X_Con_np = np.vstack(
        X_con_list
    )

    X_Rmi_np = np.vstack(
        X_rmi_list
    )

    X_Se_np = np.vstack(
        X_se_list
    )

    Y_np = np.array(
        Y_list,
        dtype=np.float32
    )


    # ======================================================
    # torch tensor
    # ======================================================

    X_Con_test = torch.from_numpy(
        X_Con_np
    )

    X_Rmi_test = torch.from_numpy(
        X_Rmi_np
    )

    X_SE_test = torch.from_numpy(
        X_Se_np
    )

    Y_test_label = torch.from_numpy(
        Y_np
    )


    matched_df = pd.DataFrame(
        matched_rows
    )


    if verbose:

        print(
            f"跳过（无 RMI）: "
            f"{len(skipped_no_rmi)}, "
            f"无 SE: "
            f"{len(skipped_no_se)}, "
            f"无 label: "
            f"{len(skipped_no_label)}"
        )


    return (
        X_Rmi_test,
        X_Con_test,
        X_SE_test,
        Y_test_label,
        matched_df
    )


# ==========================================================
# 4. 查找 model 目录中的全部 .pth 文件
# ==========================================================

def find_model_files(model_dir):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")
    files = [
        name for name in os.listdir(model_dir)
        if name.lower().endswith(".pth")
        and os.path.isfile(os.path.join(model_dir, name))
    ]
    # 自然排序：test_2.pth 排在 test_10.pth 前面。
    files.sort(key=lambda name: [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", name)
    ])
    if not files:
        raise RuntimeError(f"{model_dir} 中没有找到 .pth 模型文件")
    # 避免同名但扩展名大小写不同的模型覆盖同一个结果文件。
    stems = [os.path.splitext(name)[0] for name in files]
    if len(stems) != len(set(stems)):
        raise ValueError("模型文件去掉扩展名后存在同名项，请先重命名以避免结果覆盖")
    return files


# ==========================================================
# 5. 按 result1.py 的规则计算 Top-K 成功情况和成功率
# ==========================================================

def get_total_complexes(con_csv):
    """根据 CON_CSV 文件名确定复合物总数 N。"""
    total_by_filename = {
        "105_inter.csv": 105,
        "28_inter.csv": 28,
        "31_inter.csv": 31,
    }

    filename = os.path.basename(con_csv.replace("\\", "/"))
    if filename not in total_by_filename:
        raise ValueError(
            f"无法根据 CON_CSV 确定复合物总数：{con_csv}。"
            "请检查文件名是否为 105_inter.csv、28_inter.csv 或 31_inter.csv。"
        )

    return total_by_filename[filename]


def get_all_complex_names(con_csv):
    """从 CON_CSV 获取复合物名称。"""
    df_con = pd.read_csv(con_csv, dtype=str)
    df_con.columns = [column.strip() for column in df_con.columns]

    if "pdb_id" not in df_con.columns:
        raise ValueError("CON_CSV 文件中必须包含 pdb_id 列。")

    group_names = (
        df_con["pdb_id"]
        .astype(str)
        .str.strip()
        .str.rsplit("_lig_", n=1)
        .str[0]
    )
    all_groups = list(dict.fromkeys(group_names.tolist()))
    all_groups = [name for name in all_groups if name and name.lower() != "nan"]


    return all_groups


def evaluate_prediction_csv(csv_path, model_name, all_group_names, total_complexes):
    """对单个模型的预测 CSV 计算每个复合物的 Top-K 成功情况。"""
    df = pd.read_csv(csv_path)

    required_columns = {"pdb_id", "y_true", "y_pred"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"预测 CSV 缺少必要列：{sorted(missing)}")

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["pdb_id", "y_true", "y_pred"]).copy()

    # 与 result1.py 一致：例如 1AM0_lig_1 -> 1AM0。
    df["group_name"] = (
        df["pdb_id"]
        .astype(str)
        .str.strip()
        .str.rsplit("_lig_", n=1)
        .str[0]
    )

    # 先将 CON_CSV 中的全部复合物初始化为失败。
    # 若某个复合物没有成功生成预测结果，它仍计入总数 N，且成功状态为 0。
    success_status = {
        group_name: {k: 0 for k in TOP_K_LIST}
        for group_name in all_group_names
    }

    for group_name, group_df in df.groupby("group_name", sort=False):
        if group_name not in success_status:
            print(
                f"[WARNING] {model_name} 中的复合物 {group_name} "
                "未出现在 CON_CSV 中，已忽略。"
            )
            continue

        # y_pred 越小排名越靠前，与 result1.py 一致。
        group_df = group_df.sort_values(
            by="y_pred", ascending=True
        ).reset_index(drop=True)
        for k in TOP_K_LIST:
            top_k_df = group_df.head(k)
            is_success = (top_k_df["y_true"] < RMSD_THRESHOLD).any()
            success_status[group_name][k] = int(is_success)

    # Sheet1：每个模型的成功复合物数量和成功率。
    summary = {
        "model": model_name,
        "N": total_complexes,
    }

    for k in TOP_K_LIST:
        success_count = sum(
            success_status[group_name][k]
            for group_name in all_group_names
        )
        success_rate = success_count / total_complexes * 100.0
        summary[f"Top{k}_success_count"] = success_count
        summary[f"Top{k}_success_rate(%)"] = round(success_rate, 2)

    # Sheet2：逐个复合物记录 Top-K 是否成功（1=成功，0=失败）。
    detail_rows = []
    for group_name in all_group_names:
        row = {
            "model": model_name,
            "complex_id": group_name,
        }
        for k in TOP_K_LIST:
            row[f"Top{k}"] = success_status[group_name][k]
        detail_rows.append(row)

    return summary, detail_rows


def save_success_workbook(prediction_files, con_csv, output_file):
    """将所有模型的成功情况和成功率保存到同一个 Excel 文件。"""
    total_complexes = get_total_complexes(con_csv)
    all_group_names = get_all_complex_names(con_csv)

    summary_rows = []
    detail_rows = []

    for model_name, csv_path in prediction_files:
        try:
            summary, details = evaluate_prediction_csv(
                csv_path=csv_path,
                model_name=model_name,
                all_group_names=all_group_names,
                total_complexes=total_complexes,
            )
            summary_rows.append(summary)
            detail_rows.extend(details)

            print(
                f"[EVALUATED] {model_name}: "
                f"Top1={summary['Top1_success_rate(%)']:.2f}%, "
                f"Top20={summary['Top20_success_rate(%)']:.2f}%"
            )
        except Exception as e:
            print(f"[EVALUATION ERROR] {model_name}: {e}")

    summary_columns = ["model", "N"]
    for k in TOP_K_LIST:
        summary_columns.extend([
            f"Top{k}_success_count",
            f"Top{k}_success_rate(%)",
        ])

    detail_columns = ["model", "complex_id"] + [
        f"Top{k}" for k in TOP_K_LIST
    ]

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    detail_df = pd.DataFrame(detail_rows, columns=detail_columns)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Sheet1", index=False)
        detail_df.to_excel(writer, sheet_name="Sheet2", index=False)

    print(f"[OK] 成功情况和成功率已保存：{output_file}")
    print(f"     复合物总数 N={total_complexes}")


# ==========================================================
# 6. 主函数
# ==========================================================

def main():

    print("=" * 70)
    print(
        f"Device: {DEVICE}"
    )
    print("=" * 70)


    # ======================================================
    # 测试数据只加载一次
    # ======================================================

    (
        X_Rmi_test,
        X_Con_test,
        X_SE_test,
        Y_test_label,
        matched_df
    ) = build_inputs_from_con(

        con_csv=CON_CSV,
        rmi_csv=RMI_CSV,
        se_csv=SE_CSV,
        label_csv=LABEL_CSV,
        verbose=True

    )


    # ======================================================
    # 查找当前 model 目录中的全部 .pth，不限制文件名前缀。
    # ======================================================

    model_files = find_model_files(MODEL_DIR)


    print()
    print("=" * 70)

    print(
        f"找到 {len(model_files)} "
        f"个模型"
    )

    for filename in model_files:

        print(
            f"  {filename}"
        )

    print("=" * 70)


    # ======================================================
    # 逐个模型测试
    # ======================================================

    success_count = 0
    failed_count = 0
    prediction_files = []


    for filename in model_files:

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            os.path.splitext(filename)[0] + ".csv"
        )


        print()
        print("=" * 70)

        print(
            f"[RUN] model: "
            f"{filename}"
        )


        try:

            # ==============================================
            # 加载模型
            # ==============================================

            model = load_model(
                model_path
            )


            # ==============================================
            # 测试
            # ==============================================

            test_predictions = (
                test_model(
                    model,
                    X_Rmi_test,
                    X_Con_test,
                    X_SE_test,
                    Y_test_label,
                    Y_MEAN,
                    Y_STD
                )
            )


            # ==============================================
            # 保存结果
            # ==============================================

            results_df = pd.DataFrame({

                "pdb_id":
                    matched_df[
                        "pdb_id"
                    ].values,

                "y_true":
                    Y_test_label
                    .squeeze()
                    .cpu()
                    .numpy(),

                "y_pred":
                    test_predictions
                    .cpu()
                    .numpy()

            })


            results_df.to_csv(
                output_file,
                index=False
            )


            print(
                f"[OK] "
                f"{filename} -> "
                f"{output_file}"
            )


            success_count += 1
            prediction_files.append((filename, output_file))


            # ==============================================
            # 清理模型显存
            # ==============================================

            del model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()


        except Exception as e:

            print(
                f"[ERROR] "
                f"{filename}: "
                f"{e}"
            )

            failed_count += 1


    # ======================================================
    # 汇总所有模型的 Top-K 成功情况和成功率
    # ======================================================

    if prediction_files:
        save_success_workbook(
            prediction_files=prediction_files,
            con_csv=CON_CSV,
            output_file=SUCCESS_XLSX,
        )
    else:
        print("[WARNING] 没有成功生成预测 CSV，未生成成功率 Excel 文件。")


    # ======================================================
    # 完成
    # ======================================================

    print()
    print("=" * 70)
    print("[DONE]")



if __name__ == "__main__":
    main()

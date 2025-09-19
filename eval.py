import matplotlib.pyplot as plt
import os
import json
import torch
import pandas as pd
import numpy as np
from exp import compute_metrics, get_metric_names, create_splits_scenes, get_exclusive_list, clip_veh_route
from nuscenes.nuscenes import NuScenes
#from nuscenes.utils.splits import create_splits_scenes

# -------------------------------------------------
# New: multi-method ablation plot
# -------------------------------------------------
def plot_ablation(data_dict, x_labels, output_path="output.jpg",
                  dpi=300, y_label="", max_error=1.0, min_error=0.0):
    """
    Draw multi-line ablation chart for several methods.

    Args:
        data_dict (dict[str, list[float]]): {method_name: y_values}.
        x_labels  (list[str]):             labels for each x position.
        output_path (str):                 file to save.
        dpi (int):                         figure resolution.
        y_label (str):                     optional y-axis label.
    """
    x = list(range(len(x_labels)))
    fig, ax = plt.subplots(figsize=(6, 4))

    # plot each method
    for method, y_vals in data_dict.items():
        ax.plot(x, y_vals, linewidth=1, marker='o', markersize=2, label=method)

    # axis limits with margin
    all_vals = [v for vals in data_dict.values() for v in vals]
    min_y = min_error
    max_y = max_error
    y_margin = (max_y - min_y) * 0.05 if max_y != min_y else 0.05
    x_margin = 0.3
    ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    # ticks / labels
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    # cosmetic
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

# ------------------------
# Draw Pilot vs OLRA
# ------------------------
def plot_pilot_vs_olra(data, x_labels, output_path="output.jpg", dpi=300, max_error=1.0, min_error=0.0):
    """
    data: list of (pilot_y, olra_y)
    x_labels: list of str
    """
    x = list(range(len(data)))
    pilot_y = [d[0] for d in data]
    olra_y = [d[1] for d in data]
    min_y = min_error
    max_y = max_error
    # 動態邊界留白
    x_margin = 0.3
    y_margin = (max_y - min_y) * 0.05

    fig, ax = plt.subplots(figsize=(6, 4))
    # 線與點
    ax.plot(x, pilot_y, linewidth=1, label='pilot', marker='o', markersize=2)
    ax.plot(x, olra_y, linewidth=1, label='olra_osm', marker='o', markersize=2)

    # 設定範圍，加入邊界
    ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    # 只顯示底部軸
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('data', min_y - y_margin))
    ax.xaxis.set_ticks_position('bottom')

    # 標籤與刻度
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_yticks(np.linspace(min_y, max_y, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(min_y, max_y, 6)], fontsize=12)

    # 移除額外間距
    plt.tight_layout()

    # 標示軸箭頭
    ax.annotate('', xy=(max(x) + x_margin, min_y - y_margin), xytext=(min(x) - x_margin, min_y - y_margin),
                arrowprops=dict(arrowstyle='->', linewidth=1.5))
    ax.annotate('', xy=(min(x) - x_margin, max_y + y_margin), xytext=(min(x) - x_margin, min_y - y_margin),
                arrowprops=dict(arrowstyle='->', linewidth=1.5))

    ax.legend(loc='best', fontsize=9)
    plt.savefig(output_path, dpi=dpi)

def ensure_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [ensure_list(i) for i in x]
    return x

def safe_mean(df, col):
    valid_values = df.loc[df[col] != -1, col]
    if valid_values.empty:
        return -1
    return valid_values.mean()

def safe_mean_list(data_list):
    return np.mean(data_list) if len(data_list) > 0 else 0

# -------------------------------------------------
# Refactored: dump_figures
# -------------------------------------------------
def dump_figures(rows):
    """
    Aggregate per-scene metrics and produce figures.

    Args:
        rows (list[list]): summary_metrics_rows collected in main().
    """
    num_intvs     = 4
    interval_lbls = ['0-10', '10-20', '20-30', '30-40']
    strictnesses  = ['strict', 'moderate', 'loose']
    methods       = ['olra', 'olra_osm', 'olra_wo_dir', 'sensor', 'openpilot', 'raw']

    # create empty containers
    hit_rates_list = {
        s: {m: [[] for _ in range(num_intvs)] for m in methods}
        for s in strictnesses
    }
    euclidean_error_list = {m: [[] for _ in range(num_intvs)]
                            for m in methods}

    # indices of metrics in each row (offset after first four columns)
    STRICT_BASE   = 0
    MODERATE_BASE = 5
    LOOSE_BASE    = 10
    EUC_BASE      = 15

    # -------------------------------------------------
    # collect raw values
    # -------------------------------------------------
    for summary in rows:
        method  = summary[0]
        if method not in methods:
            continue
        metrics = summary[5:]                     # first 5 cols are meta
        for i in range(num_intvs):
            # strict / moderate / loose hit-rates
            if metrics[STRICT_BASE + i] != -1:
                hit_rates_list['strict'][method][i].append(
                    metrics[STRICT_BASE + i])
            if metrics[MODERATE_BASE + i] != -1:
                hit_rates_list['moderate'][method][i].append(
                    metrics[MODERATE_BASE + i])
            if metrics[LOOSE_BASE + i] != -1:
                hit_rates_list['loose'][method][i].append(
                    metrics[LOOSE_BASE + i])
            # euclidean errors (no strictness)
            if metrics[EUC_BASE + i] != -1:
                euclidean_error_list[method][i].append(
                    metrics[EUC_BASE + i])

    # -------------------------------------------------
    # compute averages
    # -------------------------------------------------
    avg_hit_rates = {
        s: {m: [safe_mean_list(hit_rates_list[s][m][k])
                for k in range(num_intvs)]
            for m in methods}
        for s in strictnesses
    }
    avg_euc_error = {
        m: [safe_mean_list(euclidean_error_list[m][k])
            for k in range(num_intvs)]
        for m in methods
    }

    # -------------------------------------------------
    # draw figures
    # -------------------------------------------------
    fig_dir = "paper/figs"
    os.makedirs(fig_dir, exist_ok=True)

    # 1) per-strictness hit-rate charts
    for st in strictnesses:
        # (a) openpilot vs olra
        pv_ol = [(avg_hit_rates[st]['openpilot'][i],
                  avg_hit_rates[st]['olra_osm'][i])
                 for i in range(num_intvs)]
        plot_pilot_vs_olra(
            pv_ol, interval_lbls,
            output_path=os.path.join(fig_dir,
                                     f"hit_rates-{st}-vs-openpilot.pdf"))

        # (b) ablation: olra, olra_osm, sensor, raw
        data_dict = {m: avg_hit_rates[st][m]
                     for m in ['olra', 'olra_osm', 'olra_wo_dir', 'sensor', 'raw']}
        plot_ablation(
            data_dict, interval_lbls,
            output_path=os.path.join(fig_dir,
                                     f"hit_rates-{st}-ablation.pdf"),
            y_label="Hit rate")

    # 2) Euclidean-error charts
    # openpilot vs olra OSM
    pv_ol_err = [(avg_euc_error['openpilot'][i],
                  avg_euc_error['olra_osm'][i])
                 for i in range(num_intvs)]
    max_err = max(max(avg_euc_error['openpilot']),
                  max(avg_euc_error['olra_osm']))
    min_err = min(min(avg_euc_error['openpilot']),
                  min(avg_euc_error['olra_osm']))

    plot_pilot_vs_olra(
        pv_ol_err, interval_lbls,
        output_path=os.path.join(fig_dir,
                                 "euclidean-vs-openpilot.pdf"),
        max_error=max_err, min_error=min_err)

    # ablation
    err_dict = {m: avg_euc_error[m]
                for m in ['olra', 'olra_osm', 'olra_wo_dir', 'sensor', 'raw']}

    max_err = max(max(avg_euc_error['olra']),
                  max(avg_euc_error['olra_osm']),
                  max(avg_euc_error['olra_wo_dir']),
                  max(avg_euc_error['sensor']),
                  max(avg_euc_error['raw']))

    min_err = min(min(avg_euc_error['olra']),
                  min(avg_euc_error['olra_osm']),
                  min(avg_euc_error['olra_wo_dir']),
                  min(avg_euc_error['sensor']),
                  min(avg_euc_error['raw']))
    plot_ablation(
        err_dict, interval_lbls,
        output_path=os.path.join(fig_dir, "euclidean-ablation.pdf"),
        y_label="Euclidean error (m)", max_error=max_err, min_error=min_err)


if __name__ == "__main__":
    SPLIT_ROOT      = "/data/NuScene/v1.0-trainval/sweeps/CAM_FRONT/split"
    TEST_SPLIT_ROOT = "/data/NuScene/v1.0-test/sweeps/CAM_FRONT/split"
    '''
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot="/data/NuScene/v1.0-trainval_meta",
        verbose=False
    )
    splits = create_splits_scenes()
    '''
    splits = create_splits_scenes()
    METHOD_DIRS = {
        "olra":      "olra",
        "sensor":    "sensor",
        "olra_osm":  "olra_osm",
        "olra_wo_dir": "olra_wo_dir",
        "raw":       "raw_gps",
        "perfect":   "perfect_gps",
        "openpilot": "openpilot",
    }

    df_trainval = pd.read_excel("/data/NuScene/v1.0-trainval/taxonomy.xlsx")
    df_test     = pd.read_excel("/data/NuScene/v1.0-test/taxonomy.xlsx")
    df_taxonomy = pd.concat([df_trainval, df_test], ignore_index=True)
    filename_to_class = dict(zip(df_taxonomy["filename"], df_taxonomy["class"]))

    summary_metrics_rows = {"val": [], "test": []}

    metric_names = get_metric_names()
    exclusive_list = get_exclusive_list() 
    for split_name in ["val", "test"]:
        for scene_name in splits[split_name]:
            cid = scene_name.split("-")[1]
            base_split = SPLIT_ROOT if split_name == "val" else TEST_SPLIT_ROOT
            json_path = os.path.join(base_split, cid, f"{cid}_culane.json")
            if not os.path.isfile(json_path):
                continue

            # If cid are in exclusive list, skip the case
            if cid in exclusive_list:
                continue

            print(f"▶ Evaluating {split_name} / case {cid} ...")

            with open(json_path, "r") as f:
                data = json.load(f)

            en_route     = torch.tensor(data["route"],     dtype=torch.float64)
            frames       = data["frame_data"]

            # 4. 若有任何一個 JSON 檔不存在，就跳過該場景
            preds = {}
            missing = []
            for method, subdir in METHOD_DIRS.items():
                path_json = os.path.join("/artifact", split_name, subdir, cid, "predicted_routes.json")
                if not os.path.isfile(path_json):
                    missing.append(method)
                else:
                    with open(path_json, "r") as f:
                        preds[method] = json.load(f)
            if missing:
                print(f"   ⚠️ 缺少以下 JSON 檔，跳過此場景：{missing}")
                continue

            logs_dict = {m: [] for m in METHOD_DIRS}
            mets_dict = {m: [] for m in METHOD_DIRS}

            # No ground truth route at all. In this case, we set the average metrics to be -1 to indicate that it is a invalid case
            if len(data["route"]) < 2:
                for idx, fr in enumerate(frames):
                    for method in METHOD_DIRS:
                        # Use driving route as the ground truth
                        pred_route = preds[method][idx]["veh_route"]
                        metrics = [-1 for name in metric_names]
                        mets_dict[method].append(metrics)
                        log = [idx, class_name]
                        if method in ("olra", "olra_osm", "sensor", "olra_wo_dir"):
                            existed = 1 if preds["olra"][idx].get("ego_lane_existed", False) else 0
                            log.append(existed)
                        log.extend(metrics)
                        logs_dict[method].append(log)
            else:
                for idx, fr in enumerate(frames):
                    if not isinstance(fr, dict):
                        continue
                    gt_route     = preds["perfect"][idx]["veh_route"]
                    # Clip gt route for evaluation
                    gt_route = clip_veh_route(torch.tensor(gt_route)).tolist()
                    filename   = os.path.basename(fr["filename"])
                    class_name = filename_to_class.get(filename, "unknown")
    
                    skip_this_frame = False
                    if len(gt_route) == 0:
                        skip_this_frame = True

                    for method in METHOD_DIRS:
                        start = preds[method][idx]["start"]
                        if start == False:
                            skip_this_frame = True

                    if skip_this_frame == True:
                        continue
                    
                    for method in METHOD_DIRS:
                        pred_route = preds[method][idx]["veh_route"]
                        # Clip pred route for evaluation
                        pred_route = clip_veh_route(torch.tensor(pred_route)).tolist()
                        metrics = compute_metrics(gt_route, pred_route)
                        mets_dict[method].append(metrics)
                        log = [idx, class_name]
                        if method in ("olra", "olra_osm", "sensor", "olra_wo_dir"):
                            existed = 1 if preds["olra"][idx].get("ego_lane_existed", False) else 0
                            log.append(existed)
                        log.extend(metrics)
                        logs_dict[method].append(log)

            # Dump reports
            for method, subdir in METHOD_DIRS.items():
                columns = ["frame_index", "class"]
                if method in ("olra", "olra_osm", "sensor", "olra_wo_dir"):
                    columns.append("ego_lane_existed")
                columns.extend(metric_names)
                df_metrics = pd.DataFrame(
                    logs_dict[method],
                    columns=columns
                )
                out_dir = os.path.join("/artifact", split_name, subdir, cid)
                os.makedirs(out_dir, exist_ok=True)
                df_metrics.to_excel(
                    os.path.join(out_dir, "per_frame_metrics.xlsx"),
                    index=False
                )

            # === ego-lane 出現比例（以 OLRA JSON 為準） ===
            ego_exists_total = sum(1 for i in range(len(frames))
                                   if preds["olra"][i].get("ego_lane_existed", False))
            ego_lane_ratio_scene = (ego_exists_total / len(frames)) if len(frames) > 0 else 0.0

            for method in METHOD_DIRS:
                loss_logs = logs_dict[method]
                met_list  = mets_dict[method]
                if not met_list:
                    continue

                # collect unique classes
                classes = {row[1] for row in loss_logs}
                class_str = ",".join(sorted(classes))

                df_met = pd.DataFrame(
                    met_list,
                    columns=metric_names
                )
                total_frames = len(df_met)
        
                mean_hit_rate_strict_0_10 = safe_mean(df_met, "hit_rate@0.5_0_10")
                mean_hit_rate_strict_10_20 = safe_mean(df_met, "hit_rate@0.5_10_20")
                mean_hit_rate_strict_20_30 = safe_mean(df_met, "hit_rate@0.5_20_30")
                mean_hit_rate_strict_30_40 = safe_mean(df_met, "hit_rate@0.5_30_40")
                mean_hit_rate_strict = safe_mean(df_met, "hit_rate@0.5")
                
                mean_hit_rate_moderate_0_10 = safe_mean(df_met, "hit_rate@1.0_0_10")
                mean_hit_rate_moderate_10_20 = safe_mean(df_met, "hit_rate@1.0_10_20")
                mean_hit_rate_moderate_20_30 = safe_mean(df_met, "hit_rate@1.0_20_30")
                mean_hit_rate_moderate_30_40 = safe_mean(df_met, "hit_rate@1.0_30_40")
                mean_hit_rate_moderate = safe_mean(df_met, "hit_rate@1.0")
                
                mean_hit_rate_loose_0_10 = safe_mean(df_met, "hit_rate@2.0_0_10")
                mean_hit_rate_loose_10_20 = safe_mean(df_met, "hit_rate@2.0_10_20")
                mean_hit_rate_loose_20_30 = safe_mean(df_met, "hit_rate@2.0_20_30")
                mean_hit_rate_loose_30_40 = safe_mean(df_met, "hit_rate@2.0_30_40")
                mean_hit_rate_loose = safe_mean(df_met, "hit_rate@2.0")

                mean_euclidean_error_0_10 = safe_mean(df_met, "euclidean_error_0_10")
                mean_euclidean_error_10_20 = safe_mean(df_met, "euclidean_error_10_20")
                mean_euclidean_error_20_30 = safe_mean(df_met, "euclidean_error_20_30")
                mean_euclidean_error_30_40 = safe_mean(df_met, "euclidean_error_30_40")
                mean_euclidean_error = safe_mean(df_met, "euclidean_error")

                # 只有 olra / olra_osm / sensor 需要填入比例，其餘 method 用 -1
                ratio_out = ego_lane_ratio_scene if method in ("olra", "olra_osm", "sensor") else -1

                summary_metrics_rows[split_name].append([
                    method,
                    cid,
                    class_str,
                    total_frames,
                    ratio_out,
                    mean_hit_rate_strict_0_10,
                    mean_hit_rate_strict_10_20,
                    mean_hit_rate_strict_20_30,
                    mean_hit_rate_strict_30_40,
                    mean_hit_rate_strict,
                    mean_hit_rate_moderate_0_10,
                    mean_hit_rate_moderate_10_20,
                    mean_hit_rate_moderate_20_30,
                    mean_hit_rate_moderate_30_40,
                    mean_hit_rate_moderate,
                    mean_hit_rate_loose_0_10,
                    mean_hit_rate_loose_10_20,
                    mean_hit_rate_loose_20_30,
                    mean_hit_rate_loose_30_40,
                    mean_hit_rate_loose,
                    mean_euclidean_error_0_10,
                    mean_euclidean_error_10_20,
                    mean_euclidean_error_20_30,
                    mean_euclidean_error_30_40,
                    mean_euclidean_error
                ])

            print(f"   ✅ {split_name}/{cid} evaluation 完成，總計 {len(frames)} frames")

    for split_name in ["val", "test"]:
        rows = summary_metrics_rows[split_name]
        if not rows:
            continue
        columns = ["method", "case_id", "class", "total_frames", "ego_lane_ratio"]
        columns.extend(metric_names)
        df_summary_metrics = pd.DataFrame(
            rows,
            columns=columns
        )
        out_dir = os.path.join("/artifact", split_name)
        os.makedirs(out_dir, exist_ok=True)
        df_summary_metrics.to_excel(os.path.join(out_dir, "all_summary_metrics.xlsx"), index=False)
        print(f"🏁 [{split_name}] all_summary_metrics saved to {os.path.join(out_dir, 'all_summary_metrics.xlsx')}")

        # Further average all scene-based metrics along the dataset, and draw figures
        # Draw figures of hit rates and euclidean error
        if split_name == "val":
            dump_figures(rows)

            # === 新增：olra_osm 趨勢圖（ego_lane_ratio vs mean_euclidean_error） ===
            fig_dir = "paper/figs"
            os.makedirs(fig_dir, exist_ok=True)

            df_osm = df_summary_metrics[df_summary_metrics["method"] == "olra_osm"][
                ["ego_lane_ratio", "euclidean_error"]
            ].copy()
            df_osm = df_osm[(df_osm["ego_lane_ratio"] >= 0) & (df_osm["euclidean_error"] >= 0)]

            x = df_osm["ego_lane_ratio"].to_numpy()
            y = df_osm["euclidean_error"].to_numpy()

            if len(x) >= 2:
                xs = np.linspace(float(x.min()), float(x.max()), 200)

                plt.figure(figsize=(6, 4))
                # 藍色散點
                plt.scatter(x, y, s=30, label="cases", color="blue")
                # 綠色線性擬合
                p1 = np.polyfit(x, y, 1)
                plt.plot(xs, np.polyval(p1, xs), label="linear fit", color="green", linewidth=2)
                # 紅色二次擬合（至少 3 點才畫）
                '''
                if len(x) >= 3:
                    p2 = np.polyfit(x, y, 2)
                    plt.plot(xs, np.polyval(p2, xs), label="quadratic fit", color="red", linewidth=2)
                '''

                plt.xlabel("Ego-lane ratio")
                plt.ylabel("Mean Euclidean error (m)")
                plt.legend(loc="best", fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "ego_lane_ratio_vs_error_olra_osm.pdf"), dpi=300)
                plt.close()


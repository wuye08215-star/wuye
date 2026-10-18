"""Li 浓度随机森林回归。依赖：numpy pandas scikit-learn matplotlib openpyxl。
运行：python Li_RF_回归版.py QSH.xlsx
复核原脚本的筛选范围：python Li_RF_回归版.py QSH.xlsx --li-max 2000 --output results_2000
仅用于位置辅助预测：追加 --coordinates；不能据此解释环境因果关系。
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不依赖 Tk 或桌面环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split

SEED = 42
# 当前文件的18个土壤字段存在完全重复/固定倍数关系，未核实前不纳入解释。
# 修正原始土壤数据并确认单位后，再明确把真实变量加入本列表。
FEATURES = ["AET", "AI", "Alpha(AET/PET)", "PET", "BIO7"]
META = ["Li", "ID", "ID_1", "E", "W", "1/0", "excel_row"]
PARAMS = {
    "max_depth": [None, 8, 16, 24],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": [0.6, 0.8, 1.0],
    "max_samples": [0.75, 1.0],
}


def metrics(observed, predicted_log):
    """同时报告自然对数尺度与原始浓度尺度；反变换不保证无偏均值。"""
    result = {}
    for scale, y, pred in [("log", np.log1p(observed), predicted_log),
                           ("Li", observed, np.expm1(predicted_log))]:
        result.update({
            f"{scale}_R2": r2_score(y, pred),
            f"{scale}_RMSE": np.sqrt(mean_squared_error(y, pred)),
            f"{scale}_MAE": mean_absolute_error(y, pred),
        })
    return result


def group_ids(values):
    return pd.factorize(pd.MultiIndex.from_arrays(np.asarray(values).T))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="?", default=Path("QSH.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("RF_results"))
    parser.add_argument("--li-max", type=float, default=None,
                        help="仅在有独立依据时设定浓度上限；默认保留全部非负观测")
    parser.add_argument("--coordinates", action="store_true",
                        help="加入 E/W 辅助位置预测；环境变量解释时建议不加")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.data)
    df.columns = df.columns.astype(str).str.strip()
    df["excel_row"] = np.arange(len(df)) + 2
    features = FEATURES + (["E", "W"] if args.coordinates else [])
    original = [c for c in df if c not in META]
    required = list(dict.fromkeys(original + features + ["E", "W", "Li"]))
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    # 不用 dropna 静默删除观测；发现异常时先修复输入数据。
    df[required] = df[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("输入存在空值或无穷值，请先核实数据。")
    if (df.Li < 0).any():
        raise ValueError("Li 出现负值，请核实缺测标记/检出限，不能直接 log1p。")
    if args.li_max is not None and (not np.isfinite(args.li_max) or args.li_max < 0):
        raise ValueError("--li-max 必须是非负有限数值。")
    audit = {"input_rows": len(df), "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
             "features": features, "Li_upper_limit": args.li_max,
             "warning": "18个土壤字段待核实；相关性与置换重要性不代表因果。"}
    if "CECS" in df and (df.CECS != 0).all():
        audit["fixed_multiples_of_CECS"] = {
            c: float((df[c] / df.CECS).iloc[0]) for c in original
            if np.allclose(df[c] / df.CECS, (df[c] / df.CECS).iloc[0])}
    if args.li_max is not None:
        audit["excluded_excel_rows"] = df.loc[df.Li > args.li_max, "excel_row"].tolist()
        df = df.loc[df.Li <= args.li_max].copy()
    df = df.reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("有效样本不足，不能按本脚本流程评价。")
    audit["used_rows"] = len(df)
    audit["zero_Li_rows"] = int((df.Li == 0).sum())
    y = np.log1p(df.Li.to_numpy())
    raw_y = df.Li.to_numpy()
    exact_groups = group_ids(df[["E", "W"]])
    # 假设 E/W 是以度表示的经纬度。1°网格仅是空间敏感性诊断；无缓冲带。
    spatial_groups = group_ids(np.floor(df[["E", "W"]].to_numpy()))
    a, b = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    spatial_cv = GroupKFold(n_splits=5)
    protocols = [("random_holdout", [(a, b)], exact_groups),
                 ("spatial_grid_1deg", spatial_cv.split(df, y, spatial_groups), spatial_groups)]
    rows, fold_rows, importance_rows, parameter_rows, prediction_tables, searches = [], [], [], [], [], []
    chart_data = {}
    for protocol, splits, groups in protocols:
        all_predictions = {name: np.full(len(df), np.nan) for name in ["original_RF", "revised_RF"]}
        for fold, (train, test) in enumerate(splits, 1):
            baseline = RandomForestRegressor(n_estimators=1000, max_depth=10,
                min_samples_leaf=3, random_state=SEED, n_jobs=4)
            baseline.fit(df.iloc[train][original], y[train])
            # 外层测试数据从不传给搜索；同一坐标/空间网格在内层也不跨折。
            search = RandomizedSearchCV(
                RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
                PARAMS, n_iter=24, scoring="neg_mean_squared_error", cv=GroupKFold(4),
                n_jobs=4, random_state=SEED, refit=True, error_score="raise")
            search.fit(df.iloc[train][features], y[train], groups=groups[train])
            revised = search.best_estimator_
            parameter_rows.append({"protocol": protocol, "fold": fold,
                "inner_CV_log_RMSE": np.sqrt(-search.best_score_), **search.best_params_})
            cv = pd.DataFrame(search.cv_results_)
            cv.insert(0, "fold", fold)
            cv.insert(0, "protocol", protocol)
            searches.append(cv)
            for name, model, columns in [("original_RF", baseline, original), ("revised_RF", revised, features)]:
                prediction = model.predict(df.iloc[test][columns])
                all_predictions[name][test] = prediction
                for subset, indices in [("train", train), ("test", test)]:
                    pred = model.predict(df.iloc[indices][columns])
                    fold_rows.append({"protocol": protocol, "fold": fold, "model": name,
                        "subset": subset, "n": len(indices), **metrics(raw_y[indices], pred)})
            importance = permutation_importance(revised, df.iloc[test][features], y[test],
                scoring="r2", n_repeats=20, random_state=SEED, n_jobs=1)
            for i, feature in enumerate(features):
                importance_rows.append({"protocol": protocol, "fold": fold, "feature": feature,
                    "mean_R2_drop": importance.importances_mean[i],
                    "shuffle_std": importance.importances_std[i]})
            print(f"{protocol} fold {fold} complete", flush=True)
        valid = np.isfinite(all_predictions["revised_RF"])
        # 空间评价中的每个预测均来自没有见过该网格的外层训练集。
        for name, pred in all_predictions.items():
            rows.append({"protocol": protocol, "model": name, "n_test": int(valid.sum()),
                         **metrics(raw_y[valid], pred[valid])})
        table = df.loc[valid, [c for c in ["excel_row", "ID", "ID_1", "E", "W", "Li"] if c in df]].copy()
        table.insert(0, "protocol", protocol)
        for name, pred in all_predictions.items():
            table[name + "_log_pred"] = pred[valid]
            table[name + "_Li_pred"] = np.expm1(pred[valid])
        prediction_tables.append(table)
        chart_data[protocol] = (y[valid], all_predictions["original_RF"][valid], all_predictions["revised_RF"][valid])
    results = pd.DataFrame(rows)
    # shuffle_std 是置换重复的标准差，不是置信区间；保留各折，不伪造统计显著性。
    with pd.ExcelWriter(args.output / "RF分析结果.xlsx") as writer:
        results.to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="fold_metrics", index=False)
        pd.DataFrame(parameter_rows).to_excel(writer, sheet_name="parameters", index=False)
        pd.DataFrame(importance_rows).to_excel(writer, sheet_name="importance_by_fold", index=False)
        pd.concat(prediction_tables, ignore_index=True).to_excel(writer, sheet_name="predictions", index=False)
    pd.concat(searches, ignore_index=True).to_csv(args.output / "training_CV_search.csv", index=False)
    (args.output / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    results.to_csv(args.output / "metrics.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for column, (protocol, (obs, old, new)) in enumerate(chart_data.items()):
        for row, (label, pred) in enumerate([("Original RF", old), ("Revised RF", new)]):
            ax = axes[row, column]
            ax.scatter(obs, pred, s=14, alpha=0.55)
            low, high = min(obs.min(), pred.min()), max(obs.max(), pred.max())
            ax.plot([low, high], [low, high], "r--", linewidth=1)
            ax.set(xlabel="Observed ln(1+Li)", ylabel="Predicted ln(1+Li)",
                   title=f"{label}: {protocol}\nR2={r2_score(obs, pred):.3f}", xlim=(low, high), ylim=(low, high))
    fig.tight_layout()
    fig.savefig(args.output / "RF_validation.png", dpi=180)
    plt.close(fig)
    print(results.to_string(index=False))
    print("输出目录：", args.output.resolve())


if __name__ == "__main__":
    main()

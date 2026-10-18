"""按论文阈值预测地下水 Li>30 μg/L 的概率。
运行：python Li_RF_分类版.py QSH.xlsx --output RF_classification_results
目标、评价尺度与 Li_RF_回归版.py 的连续浓度回归不同。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupKFold, RandomizedSearchCV,
    StratifiedGroupKFold, train_test_split)

SEED = 42
LI_THRESHOLD = 30.0  # 论文的研究分组阈值；高 Li 严格大于 30。
FEATURES = ["AET", "AI", "Alpha(AET/PET)", "PET", "BIO7"]
PARAMS = {"max_depth": [None, 5, 8, 12], "min_samples_leaf": [1, 3, 5, 10, 20],
          "max_features": ["sqrt", 0.6, 1.0], "max_samples": [0.75, 1.0]}


def metrics(y, probability):
    pred = (probability >= 0.5).astype(int)  # 不在测试集上挑概率截断点。
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred),
        "Balanced_accuracy": balanced_accuracy_score(y, pred),
        "ROC_AUC": roc_auc_score(y, probability),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Brier": brier_score_loss(y, probability),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def group_ids(values):
    return pd.factorize(pd.MultiIndex.from_arrays(np.asarray(values).T))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="?", default=Path("QSH.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("RF_classification_results"))
    parser.add_argument("--coordinates", action="store_true",
                        help="位置辅助预测；不要据此解释环境因果关系")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.data)
    df.columns = df.columns.astype(str).str.strip()
    df["excel_row"] = np.arange(len(df)) + 2
    features = FEATURES + (["E", "W"] if args.coordinates else [])
    meta = ["Li", "ID", "ID_1", "E", "W", "1/0", "excel_row"]
    original = [c for c in df if c not in meta]
    required = list(dict.fromkeys(original + features + ["Li", "E", "W"]))
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    df[required] = df[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[required].to_numpy(float)).all() or (df.Li < 0).any():
        raise ValueError("请先核实缺失值、无穷值或负 Li 值，不静默删除样本。")
    y = (df.Li.to_numpy() > LI_THRESHOLD).astype(int)
    # 不使用现成1/0标签，不将Li或它的派生标签放入输入。
    audit = {"n": len(df), "Li_unit": "ug/L (according to paper)",
             "high_class_definition": "Li > 30", "high_n": int(y.sum()),
             "low_n": int((y == 0).sum()), "features": features,
             "warning": "土壤属性未核实前暂不解释；概率输出尚未经外部校准验证。"}
    mismatch = pd.DataFrame()
    if "1/0" in df:
        different = pd.to_numeric(df["1/0"], errors="coerce").to_numpy() != y
        audit["old_label_mismatches"] = int(different.sum())
        mismatch = df.loc[different, ["excel_row", "Li", "1/0"]].copy()
        mismatch["correct_class_Li_gt_30"] = y[different]
    (args.output / "classification_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    exact_groups = group_ids(df[["E", "W"]])
    # 假设E/W为度单位经纬度。1°网格仅作空间敏感性诊断，无空间缓冲带。
    spatial_groups = group_ids(np.floor(df[["E", "W"]].to_numpy()))
    train, test = train_test_split(np.arange(len(df)), test_size=0.3,
                                  stratify=y, random_state=SEED)
    protocols = [("paper_70_30", [(train, test)], exact_groups),
                 ("spatial_grid_1deg", GroupKFold(5).split(df, y, spatial_groups), spatial_groups)]
    rows, fold_rows, parameter_rows, importance_rows, predictions, searches = [], [], [], [], [], []
    charts = {}
    for protocol, splits, groups in protocols:
        probs = {name: np.full(len(df), np.nan) for name in ["reference_RF23", "revised_RF5"]}
        for fold, (train, test) in enumerate(splits, 1):
            # 参考分类器与原回归代码使用相同树数/深度/叶子大小，但它是新分类任务。
            baseline = RandomForestClassifier(n_estimators=1000, max_depth=10,
                min_samples_leaf=3, random_state=SEED, n_jobs=4)
            baseline.fit(df.iloc[train][original], y[train])
            inner_cv = (StratifiedGroupKFold(10, shuffle=True, random_state=SEED)
                        if protocol == "paper_70_30" else GroupKFold(4))
            # 目标是概率判别，训练内按ROC-AUC选择；准确率另按固定0.5阈值报告。
            # 不能把ROC-AUC当作正确率，Brier越低越好，但不等于已经完成概率校准。
            search = RandomizedSearchCV(
                RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=1),
                PARAMS, n_iter=24, scoring="roc_auc", cv=inner_cv, n_jobs=4,
                random_state=SEED, refit=True, error_score="raise")
            search.fit(df.iloc[train][features], y[train], groups=groups[train])
            model = search.best_estimator_
            parameter_rows.append({"protocol": protocol, "fold": fold,
                "inner_CV_ROC_AUC": search.best_score_, **search.best_params_})
            cv = pd.DataFrame(search.cv_results_)
            cv.insert(0, "fold", fold)
            cv.insert(0, "protocol", protocol)
            searches.append(cv)
            for name, fitted, columns in [("reference_RF23", baseline, original), ("revised_RF5", model, features)]:
                probs[name][test] = fitted.predict_proba(df.iloc[test][columns])[:, 1]
                for subset, indices in [("train", train), ("test", test)]:
                    probability = fitted.predict_proba(df.iloc[indices][columns])[:, 1]
                    fold_rows.append({"protocol": protocol, "fold": fold, "model": name,
                        "subset": subset, "n": len(indices), **metrics(y[indices], probability)})
            importance = permutation_importance(model, df.iloc[test][features], y[test],
                scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=1)
            for i, feature in enumerate(features):
                importance_rows.append({"protocol": protocol, "fold": fold, "feature": feature,
                    "mean_AUC_drop": importance.importances_mean[i],
                    "shuffle_std": importance.importances_std[i]})
            print(f"{protocol} fold {fold} complete", flush=True)
        valid = np.isfinite(probs["revised_RF5"])
        for name, probability in probs.items():
            rows.append({"protocol": protocol, "model": name, "n_test": int(valid.sum()),
                         **metrics(y[valid], probability[valid])})
        table = df.loc[valid, [c for c in ["excel_row", "ID", "ID_1", "E", "W", "Li"] if c in df]].copy()
        table.insert(0, "protocol", protocol)
        table["class_Li_gt_30"] = y[valid]
        for name, probability in probs.items():
            table[name + "_high_probability"] = probability[valid]
            table[name + "_class"] = (probability[valid] >= 0.5).astype(int)
        predictions.append(table)
        charts[protocol] = (y[valid], {name: p[valid] for name, p in probs.items()})
    results = pd.DataFrame(rows)
    with pd.ExcelWriter(args.output / "RF分类结果.xlsx") as writer:
        for sheet, table in [("metrics", results), ("fold_metrics", pd.DataFrame(fold_rows)),
            ("parameters", pd.DataFrame(parameter_rows)), ("importance_by_fold", pd.DataFrame(importance_rows)),
            ("predictions", pd.concat(predictions, ignore_index=True)), ("label_mismatches", mismatch)]:
            table.to_excel(writer, sheet_name=sheet, index=False)
    results.to_csv(args.output / "classification_metrics.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(args.output / "classification_training_CV.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, (protocol, (obs, probabilities)) in enumerate(charts.items()):
        for name, probability in probabilities.items():
            fpr, tpr, _ = roc_curve(obs, probability)
            axes[row, 0].plot(fpr, tpr, label=f"{name}: AUC={roc_auc_score(obs, probability):.3f}")
        axes[row, 0].plot([0, 1], [0, 1], "k--", linewidth=0.7)
        axes[row, 0].set(xlabel="False positive rate", ylabel="True positive rate", title=protocol)
        axes[row, 0].legend(loc="lower right")
        ConfusionMatrixDisplay.from_predictions(obs, probabilities["revised_RF5"] >= 0.5,
            labels=[0, 1], display_labels=["Li<=30", "Li>30"], cmap="Blues", colorbar=False, ax=axes[row, 1])
        axes[row, 1].set_title("Revised RF, probability cutoff=0.5")
    fig.tight_layout()
    fig.savefig(args.output / "classification_validation.png", dpi=180)
    plt.close(fig)
    print(results.to_string(index=False))
    print("输出目录：", args.output.resolve())


if __name__ == "__main__":
    main()

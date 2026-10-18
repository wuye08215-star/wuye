"""RF训练CV选中变量子集的试验版；保留完整测试样本，结论见变量筛选报告。
最终预测与重要性均采用Li非负下限；未截断预测同时保留。
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 保存图片，不依赖桌面界面
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ( mean_squared_error, r2_score, mean_absolute_error)
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def nonnegative_r2(estimator, X, y):
    # Li>=0，因此ln(1+Li)>=0；重要性评价也采用同一预测规则。
    return r2_score(y, np.maximum(estimator.predict(X), 0.0))


# 1. 数据清洗
def clean_data_final(df, features, target):
    data = df[features + [target]].copy()
    data = data.astype(str).apply( lambda x: x.str.strip())
    chars_to_remove = [
        "[", "]", "(", ")", "{", "}", '"', "'",
        "［", "］", "（", "）", "｛", "｝",
        "“", "”", "‘", "’",
        "「", "」", "『", "』", "【", "】"
    ]

    for ch in chars_to_remove:
        data = data.replace( ch, "", regex=False)
    data = data.replace( r"[\u200b\u200e\u202a\u202c\ufeff]", "", regex=True)

    data = data.apply( pd.to_numeric, errors="coerce")
    before = len(data)
    data = data.dropna()
    after = len(data)
    X = data[features]
    y = data[target]
    return X, y, before, after

# 2. 拟合散点图
def plot_fit_scatter( y_true, y_pred, save_path="RF模型拟合散点图.png"):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt( mean_squared_error( y_true,y_pred) )
    mae = mean_absolute_error( y_true, y_pred)
    nse = 1 - np.sum( (y_true - y_pred) ** 2 ) / np.sum( (y_true - np.mean(y_true)) ** 2 )
    plt.figure(figsize=(6, 6))

    plt.scatter( y_true, y_pred,  alpha=0.6, edgecolors="k", linewidth=0.5 )
    min_val = min( np.min(y_true), np.min(y_pred))
    max_val = max( np.max(y_true), np.max(y_pred))

    plt.plot( [min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("实测值 log(Li+1)")
    plt.ylabel("预测值 log(Li+1)")
    plt.title("Random Forest模型拟合散点图")

    textstr = (
        f"$R^2={r2:.4f}$\n"
        f"$RMSE={rmse:.4f}$\n"
        f"$MAE={mae:.4f}$\n"
        f"$NSE={nse:.4f}$"
    )

    plt.text( 0.05, 0.95, textstr, transform=plt.gca().transAxes, verticalalignment="top", fontsize=11,
        bbox=dict( boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.grid( True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig( save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return r2, rmse, mae, nse

# 3. RF变量重要性
def importance_analysis( model, X_test, y_test, features, feature_groups):
    result = permutation_importance( model, X_test, y_test, n_repeats=20, random_state=42, scoring=nonnegative_r2)
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": result.importances_mean,
        "Shuffle_STD": result.importances_std,
        "Equivalent_fields": [", ".join(feature_groups[f]) for f in features]
    })
    importance_df = importance_df.sort_values( "Importance", ascending=False)
    print("\nRF变量重要性：")
    print(importance_df)

    importance_df.to_excel( "RF变量重要性.xlsx", index=False)
    top = importance_df.head(15).sort_values( "Importance", ascending=True)

    plt.figure(figsize=(8, 6))
    labels = [f"{f} (shared group)" if len(feature_groups[f]) > 1 else f
              for f in top["Feature"]]
    plt.barh(labels, top["Importance"], xerr=top["Shuffle_STD"], capsize=3)

    plt.xlabel("Decrease in test R² after shuffling")
    plt.ylabel("Environmental Factor")
    plt.title("Random Forest Variable Importance")
    plt.tight_layout()

    plt.savefig( "RF变量重要性.png", dpi=300, bbox_inches="tight")
    plt.close()

# 4. 主程序
def run_model(df):
    target = "Li"
    remove_columns = [
        "Li",
        "ID",
        "ID_1",
        "E",
        "W",
        "1/0"
    ]

    features = [
        col for col in df.columns
        if col not in remove_columns
    ]

    print("输入变量数量：", len(features))
    print(features)

    X_df, y, before, after = clean_data_final( df, features, target)
    print("因缺失值/非数值删除样本：", before - after)

    # 删除极端Li
    before_outlier = len(y)
    mask = y <= 2000
    X_df = X_df.loc[mask]
    y = y.loc[mask]
    print( "删除Li>2000极端样本：", before_outlier - len(y))

    # log转换
    y = np.log1p(y)

    X = X_df.values.astype(float)
    y = y.values.astype(float)

    # 划分和原代码一致，附带原Excel行号以便检查预测。
    X_train, X_test, y_train, y_test, rows_train, rows_test = train_test_split(
        X, y, X_df.index.to_numpy() + 2, test_size=0.2, random_state=42)

    # 仅合并训练X中完全线性重复的字段，不用y或测试集筛选变量。
    # 当前23列包含6个不同的数值方向；合并后保留全部已有数值信息。
    corr = np.abs(np.corrcoef(X_train, rowvar=False))
    kept = []
    field_map = []
    feature_groups = {}
    for j, name in enumerate(features):
        representative = next((k for k in kept
            if np.isclose(corr[j, k], 1.0, rtol=0, atol=1e-12)), None)
        if representative is None:
            kept.append(j)
            representative = j
            feature_groups[name] = []
        kept_name = features[representative]
        feature_groups[kept_name].append(name)
        field_map.append({"Original_field": name, "Retained_field": kept_name})
    # 仅按训练内五折CV选中的固定子集；不依据测试结果挑列。
    selected_features = ['AET', 'AI', 'Alpha(AET/PET)', 'BIO7']
    kept = [j for j in kept if features[j] in selected_features]
    pd.DataFrame(field_map).to_excel("RF重复字段对应表.xlsx", index=False)
    # 固定内存布局，减少非凸MLP求解的数值路径差异。
    X_train = np.ascontiguousarray(X_train[:, kept])
    X_test = np.ascontiguousarray(X_test[:, kept])
    features = [features[k] for k in kept]
    print("合并完全线性重复字段后输入数量：", len(features), features)

    # 按训练内固定5折R²选出的参数；未按测试集R²挑参数。
    # ccp_alpha用于剪枝，max_depth限制复杂度，max_features降低树间相关。
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=1,
        max_features=0.3,
        ccp_alpha=0.001,
        random_state=42,
        n_jobs=-1
    )

    model.fit( X_train, y_train)

    # 画RF中的一棵树
    tree = model.estimators_[0]
    plt.figure(figsize=(25, 12))
    plot_tree(
        tree,
        feature_names=features,
        filled=True,
        rounded=True,
        max_depth=2,
        fontsize=10
    )

    plt.title( "Random Forest Decision Tree Structure")
    plt.savefig( "RF决策树结构.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 预测
    raw_pred = model.predict(X_test)
    # 固定物理下限，不用测试数据挑阈值；原始预测也保存以便审查。
    y_pred = np.maximum(raw_pred, 0.0)
    pd.DataFrame({
        "Excel_row": rows_test,
        "Observed_ln1p_Li": y_test,
        "Predicted_ln1p_Li": y_pred,
        "Raw_predicted_ln1p_Li": raw_pred,
        "Observed_Li": np.expm1(y_test),
        "Predicted_Li": np.expm1(y_pred)
    }).to_excel("RF测试集预测.xlsx", index=False)
    r2,rmse, mae, nse = plot_fit_scatter( y_test, y_pred)

    print("\nRandom Forest模型结果")
    print("R2 =", r2)
    print("RMSE =", rmse)
    print("MAE =", mae)
    print("NSE =", nse)

    importance_analysis(
        model,
        X_test,
        y_test,
        features,
        feature_groups
    )

    result_df = pd.DataFrame({
        "Model": ["Random Forest"],
        "R2": [r2],
        "RMSE": [rmse],
        "MAE": [mae],
        "NSE": [nse],
        "N_clipped_to_zero": [int((raw_pred < 0).sum())],
        "Train_R2": [r2_score(y_train, np.maximum(model.predict(X_train), 0.0))],
        "N_train": [len(y_train)],
        "N_test": [len(y_test)]
    })

    result_df.to_excel( "RF模型评价结果.xlsx", index=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="以原RF代码为基础的小改回归版本")
    parser.add_argument("data", nargs="?", default="QSH.xlsx")
    parser.add_argument("--output", default="RF_results_minimal")
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    os.chdir(output_path)
    df = pd.read_excel(data_path)
    run_model(df)
    print("Random Forest模型运行完成，结果目录：", output_path)

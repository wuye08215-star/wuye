import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import ( mean_squared_error, r2_score, mean_absolute_error)
from sklearn.inspection import permutation_importance

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

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
def plot_fit_scatter( y_true, y_pred, save_path="SVR模型拟合散点图.png"):
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
    plt.title("SVR模型拟合散点图")

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
    plt.show()

    return r2, rmse, mae, nse

# 3. SVR变量重要性
def importance_analysis( model, X_test, y_test, features):
    result = permutation_importance( model, X_test, y_test, n_repeats=20, random_state=42, scoring="r2")
    importance_df = pd.DataFrame({ "Feature": features, "Importance": result.importances_mean})
    importance_df = importance_df.sort_values( "Importance", ascending=False)
    print("\nSVR变量重要性：")
    print(importance_df)

    importance_df.to_excel( "SVR变量重要性.xlsx", index=False)
    top = importance_df.head(15).sort_values( "Importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh( top["Feature"], top["Importance"])

    plt.xlabel("Permutation Importance")
    plt.ylabel("Environmental Factor")
    plt.title("SVR Variable Importance")
    plt.tight_layout()

    plt.savefig( "SVR变量重要性.png", dpi=300, bbox_inches="tight")
    plt.show()

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

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    model = Pipeline([( "scaler", StandardScaler()),( "svr", SVR( kernel="rbf", C=100, gamma=0.1, epsilon=0.05))])
    model.fit( X_train, y_train)

    # 预测
    y_pred = model.predict( X_test)
    r2,rmse, mae, nse = plot_fit_scatter( y_test, y_pred)

    print("\nSVR模型结果")
    print("R2 =", r2)
    print("RMSE =", rmse)
    print("MAE =", mae)
    print("NSE =", nse)

    importance_analysis(
        model,
        X_test,
        y_test,
        features
    )

    result_df = pd.DataFrame({
        "Model": ["SVR"],
        "R2": [r2],
        "RMSE": [rmse],
        "MAE": [mae],
        "NSE": [nse]
    })

    result_df.to_excel( "SVR模型评价结果.xlsx", index=False)
if __name__ == "__main__":
    df = pd.read_excel( r"C:/Users/Administrator/PycharmProjects/pythonProject4/QSH.xlsx")
    run_model(df)
    print("SVR模型运行完成")
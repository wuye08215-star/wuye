# 其他机器学习方法：新代码与结果

新增XGBoost、高斯过程回归、样条加性回归，以及RF/GP平均。不是所有新方法都更好：固定测试R²分别0.4982、0.5440、0.4939、0.5506；当前四因子RF为0.5538。GP和组合训练CV更高，仍需要独立数据比较，不认定RF在所有新数据上更优。

## 直接运行

将QSH.xlsx放在解压根目录，在该目录打开终端：

```bash
python -m pip install -r requirements_other_methods.txt
python code/Li_other_methods.py QSH.xlsx --method xgboost
python code/Li_other_methods.py QSH.xlsx --method gpr
python code/Li_other_methods.py QSH.xlsx --method spline
python code/Li_other_methods.py QSH.xlsx --method rf_gp
```

推荐Python3.12；专用CPU版XGBoost依赖适用于Windows/Linux。可使用`--output 结果目录`指定保存位置。每种方法输出评价Excel、逐行预测、重新评价的置换重要性与两张图片。

新代码复用同目录RF_modified.py的原始清洗函数，两个py文件必须放在同一code目录。专用代码ZIP已经同时包含。4因子为AET、AI、Alpha(AET/PET)、BIO7；原Li≤2000筛选、ln(1+Li)、随机种子42和273条测试样本保持一致，不新增删样本。

训练内128组配置、640个验证分数及11种组合比例完整保留。参数只按训练CV选，最终三种新方法均选中4因子；组合权重为RF30%、GP70%。本轮训练CV受此前输入筛选与本次调参影响，固定测试也已经过多轮开发查看，都不能替代独立数据验证。置换重要性只表示预测依赖，不是因果影响。

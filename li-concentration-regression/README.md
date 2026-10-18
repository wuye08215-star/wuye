# Li浓度回归：四方法组会记录（当前版本）

以用户原始RF、SVR、LightGBM、MLP代码和QSH数据为准。这里保存本次有效的连续浓度回归研究，旧方案已废弃。

## 下载

- [8种方案详细表总文件（9个工作表，内嵌全部图册）](eight_model_tables/All_8_models.xlsx)
- [8份模型详表＋1份总对比，以及综合文件打包](eight_model_tables/Eight_models_all_tables.zip)

- [组会Word报告（25页，含新方法及各轮补充）](Li_group_meeting_report.docx)
- [PDF阅读版](Li_group_meeting_report.pdf)
- [最初四方法结果总表Excel](group_meeting_results.xlsx)
- [全部代码、报告和结果ZIP](full_package.zip)
- [仅报告与研究结果ZIP](results_only.zip)

在GitHub打开二进制文件后，可点击“Download raw file”保存到电脑。代码保持原始结构，修改位置见diffs/，各版本均已实跑。

## 最初四方法修改结果（变量筛选前）

| 模型 | 原始R² | 最终R² | ΔR² |
|---|---:|---:|---:|
| RF | 0.5390 | 0.5403 | +0.0013 |
| SVR | 0.4680 | 0.5135 | +0.0456 |
| LightGBM | 0.5396 | 0.4851 | -0.0545 |
| MLP | 0.3562 | 0.5007 | +0.1445 |

指标尺度为ln(1+Li)，不是分类准确率。RF基本不变；SVR有一定改善；LightGBM本轮试验退化，不建议替换原版。

**MLP总增益+0.1445必须拆开汇报：训练修改仅+0.0171，非负下限修正约+0.1273，主要来自一条负预测。不能把后处理收益全归因于调参。** 原始和未截断预测均有记录。原SVR仅加下限就可达R²=0.4979，完整修改版为0.5135。

## 报告记录了什么

1. 原代码与上传Excel复现、数据和评价规则。
2. 588组训练内搜索、每种算法的参数前后对照。
3. 正则化、去重复表示、迭代求解、未收敛警告与C/F数值路径诊断。
4. 非负浓度约束的单独消融，包括原始负预测所在Excel行号。
5. 测试集升降、CV波动、统一组置换重要性和RF去一组因子重训练。
6. 失败结果、解释限制与可直接用于组会的口头汇报简稿。

## 运行

推荐Python3.12。将你自己的QSH.xlsx放在解压目录，再执行：

```bash
python -m pip install -r requirements.txt
python code/RF_modified.py QSH.xlsx --output results/RF
python code/SVR_modified.py QSH.xlsx --output results/SVR
python code/LightGBM_modified.py QSH.xlsx --output results/LightGBM
python code/MLP_modified.py QSH.xlsx --output results/MLP
```

LightGBM试验版保留为研究记录，当前测试效果不如原始版本，原版在originals/。树图另需要系统Graphviz程序dot。MLP对计算库和非凸数值路径敏感，本次固定C连续布局，并保留环境和过程记录。

## 结果解释

当前23列存在18个完全线性重复字段，只能识别共享信息，不能分别给出18项真实物理作用。气候因子之间也高度相关。置换分数是预测依赖程度，不是因果效应或可相加的贡献率；MLP置换分数偏大反映外推敏感，不应解释成真实影响倍数。

同输入不同Li的实例、源文件哈希、固定行号预测、逐折分数和警告记录均保存在Excel及audit_logs/。不通过换测试集、删除高误差测试点或隐藏负面结果提高分数。

## 追加：迭代与异常样本补充检验

本轮63组（含对照）及11个RF/SVR权重，详见 [5页补充Word](additional_experiments/R2_additional_experiments.docx)、[补充PDF](additional_experiments/R2_additional_experiments.pdf)、[全部补充数值Excel](additional_experiments/additional_results.xlsx)。已追加到主组会报告。

- 原RF实跑R²约0.5390，上一版修改RF约0.5403。
- 本轮最高观测R²=0.5442，但依赖额外删除训练尾部10条；仅作敏感性分析，未确认这些记录有错。最小改动脚本为 `code/RF_training_tail_sensitivity.py`，实跑验证通过。
- 不新增删样本的RF/SVR各50%平均R²=0.5432。
- 更多迭代、IsolationForest、缩尾、Huber都未带来明显突破。测试集273条从未新增删减。
- 目前不能认定已到上限，也不能宣称穷尽所有方法；小幅增益仍需独立数据确认。

运行删尾部敏感性脚本：

```bash
python code/RF_training_tail_sensitivity.py QSH.xlsx --output results/RF_tail_sensitivity
```

这一脚本不替换主分析。删样本行号和全部测试预测记录在 `additional_experiments/RF_tail_sensitivity_results/`。

## 追加：变量数量与组合筛选

已检验原23列、训练内排名前6/7列及6个非重复信息组的全部63个非空子集，共67组RF方案、335个逐折分数。

按训练CV选中AET、AI、Alpha(AET/PET)、BIO7四列，CV R²=0.4503，完整测试R²=0.5538；相同修改RF参数下6组为0.5403。训练与测试均未新增删样本。原RF参数下前6/7列分别为0.5403/0.5393。

- [3页变量筛选Word](feature_selection/Feature_selection_report.docx) / [PDF](feature_selection/Feature_selection_report.pdf)
- [变量筛选完整Excel](feature_selection/feature_selection_results.xlsx)
- 最小改动候选代码：`code/RF_selected_subset.py`，已在原QSH实跑，图表与逐行预测在`feature_selection/RF_selected_subset_results/`。

四列为简化预测候选，6组保留作对照；原始23列数据保留。该选择不证明PET及土壤因子没有物理作用，仅适用于本次RF与内部验证。

```bash
python code/RF_selected_subset.py QSH.xlsx --output results/RF_selected_subset
```

## 追加：其他机器学习方法代码

新增XGBoost、高斯过程回归、样条加性回归，并检查RF/GP组合；固定测试R²分别为0.4982、0.5440、0.4939、0.5506。本轮均未超过已有四因子RF的0.5538；GP和组合的训练CV更高，仍需独立验证，不据此认定谁在新样本上一定更好。

- [新方法一页Excel简表](Other_methods_brief.xlsx)
- [可直接运行的新方法代码ZIP（含清洗依赖与说明）](Other_methods_code.zip)
- [完整新方法训练记录、预测及图片](other_methods/)

在完整包根目录安装并运行：

```bash
python -m pip install -r requirements_other_methods.txt
python code/Li_other_methods.py QSH.xlsx --method gpr
```

可将gpr换成xgboost、spline或rf_gp。依赖CPU版XGBoost，已在本云环境验证；Windows/Linux推荐Python3.12。新的安装说明已存入云环境install_script草稿，尚未发布环境快照。

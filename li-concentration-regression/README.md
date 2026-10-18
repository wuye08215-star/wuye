# Li浓度回归：四方法组会记录（当前版本）

以用户原始RF、SVR、LightGBM、MLP代码和QSH数据为准。这里保存本次有效的连续浓度回归研究，旧方案已废弃。

## 下载

- [组会Word报告（15页）](Li_group_meeting_report.docx)
- [PDF阅读版](Li_group_meeting_report.pdf)
- [结果总表Excel](group_meeting_results.xlsx)
- [全部代码、报告和结果ZIP](full_package.zip)
- [仅报告与研究结果ZIP](results_only.zip)

在GitHub打开二进制文件后，可点击“Download raw file”保存到电脑。代码保持原始结构，修改位置见diffs/，各版本均已实跑。

## 同一批273条测试样本的结果

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

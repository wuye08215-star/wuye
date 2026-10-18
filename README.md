# wuye

## Li浓度回归与因子重要性

- [8种模型详细Excel总文件（9个工作表，内嵌图册）](li-concentration-regression/eight_model_tables/All_8_models.xlsx)
- [8份详表＋1份总对比及综合文件打包下载](li-concentration-regression/eight_model_tables/Eight_models_all_tables.zip)

以原始RF、SVR、LightGBM、MLP代码和QSH数据为基础，研究连续Li浓度预测与因子预测依赖。原始数据保留，模型重要性不作为因果结论。

当前四因子RF候选保留AET、AI、Alpha(AET/PET)、BIO7，固定测试R²=0.5538。本轮新增XGBoost、高斯过程、样条加性回归及RF/GP组合，固定测试均未超过这一结果；GP及组合训练CV更高，仍需独立样本比较。测试273条完整保留，本轮未新增删样本。

- [原模型改进一页Excel](li-concentration-regression/Li_improvement_brief.xlsx)
- [新方法对照一页Excel](li-concentration-regression/Other_methods_brief.xlsx)
- [新方法可运行代码ZIP](li-concentration-regression/Other_methods_code.zip)
- [四因子RF候选代码](li-concentration-regression/code/RF_selected_subset.py)
- [完整组会Word（25页）](li-concentration-regression/Li_group_meeting_report.docx) / [PDF](li-concentration-regression/Li_group_meeting_report.pdf)
- [完整代码与研究记录](li-concentration-regression/) / [全部资料ZIP](li-concentration-regression/full_package.zip)

变量筛选、迭代与训练异常样本敏感性、正负实验结果及逐行预测均有记录。旧分类方案已废弃，历史提交仍可追溯。

# wuye

## 当前研究资料：Li浓度回归与因子重要性

本次以原始RF、SVR、LightGBM、MLP代码为准，保留连续浓度回归任务。旧方案已从当前版本移除；历史提交仍可追溯。

- [组会Word报告（20页，含补充实验）](li-concentration-regression/Li_group_meeting_report.docx)
- [PDF阅读版](li-concentration-regression/Li_group_meeting_report.pdf)
- [完整代码与研究记录](li-concentration-regression/)
- [下载全部代码与结果ZIP](li-concentration-regression/full_package.zip)

结果：RF基本持平；SVR有一定改善；MLP改善主要来自物理非负约束，报告中分开归因；LightGBM本轮改动退化，不推荐替换原版。详见报告与逐项对照。

本轮针对迭代与异常样本处理新增63组配置（含对照）和11个RF/SVR权重比较，完整测试集273条保留。删10条训练尾部记录的敏感性R²=0.5442，不新增删样本的简单平均R²=0.5432；增益均小，不代表已到上限或显著突破。

- [本轮5页补充Word](li-concentration-regression/additional_experiments/R2_additional_experiments.docx)
- [补充PDF](li-concentration-regression/additional_experiments/R2_additional_experiments.pdf)
- [补充实验Excel](li-concentration-regression/additional_experiments/additional_results.xlsx)

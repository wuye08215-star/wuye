# wuye

## 当前研究资料：Li浓度回归与因子重要性

本次以原始RF、SVR、LightGBM、MLP代码为准，保留连续浓度回归任务。旧方案已从当前版本移除；历史提交仍可追溯。

- [组会Word报告（15页）](li-concentration-regression/Li_group_meeting_report.docx)
- [PDF阅读版](li-concentration-regression/Li_group_meeting_report.pdf)
- [完整代码与研究记录](li-concentration-regression/)
- [下载全部代码与结果ZIP](li-concentration-regression/full_package.zip)

结果：RF基本持平；SVR有一定改善；MLP改善主要来自物理非负约束，报告中分开归因；LightGBM本轮改动退化，不推荐替换原版。详见报告与逐项对照。

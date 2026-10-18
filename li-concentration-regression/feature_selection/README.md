# RF变量筛选检验

比较原RF的23列与训练内排名前6/7列，以及修改RF在6个非重复信息组的全部63个子集。共67组方案、335个验证分数；训练/测试样本均未新增删除。

- [变量筛选Word报告](Feature_selection_report.docx) / [PDF](Feature_selection_report.pdf)
- [完整结果Excel](feature_selection_results.xlsx)
- `run_feature_selection.py`：安装父目录requirements.txt后运行，使用本目录清洗数据快照及原协议。前6/7列排名严格在训练折内重做。
- `selection_before_test.json`：在本轮固定测试评价前保存的选择结果。

训练CV总体选中：AET | AI | Alpha(AET/PET) | BIO7，共4组；CV R²=0.450280，测试R²=0.553812，相对6组RF变化+0.013480。选择不依据测试最高值。

最小改动候选代码在 `../code/RF_selected_subset.py`，相对当前RF修改版的差异见 `../diffs/RF_selected_subset.diff`。不新增删样本。科学解释须保留相关因子替代、字段重复和内部验证的限制；不能将被删变量解释为没有物理作用。

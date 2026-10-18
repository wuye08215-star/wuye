# R²补充实验

本轮63组配置（含对照）及11个RF/SVR权重比较。测试273条全部保留，训练规则和参数仅按训练五折CV选择。

- `R2_additional_experiments.docx` / `.pdf`：本轮独立汇报材料；同时已追加至主组会报告。
- `additional_results.xlsx`：全部对照、逐折分数、组合权重及新增删除行号。
- `protocol_frozen.json` / `selection_before_test.json`：评分与选择规则记录。
- `frozen_data.npz`：来自原QSH的清洗后数据、固定训练测试划分、源Excel行号；源文件哈希见父目录。
- `run_additional.py` / `ensemble_check.py`：复现本轮全部检验。先在本目录安装父目录requirements.txt，再执行两个脚本。推荐设置OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。运行会更新本目录的实验输出，模型缓存写入models/。

最高观测测试R²=0.5442依赖额外删除10条训练尾部记录（4条零值、6条660–977），这些样本尚未被证实为错误。不新增删样本的RF/SVR平均为0.5432。二者增益都小，不能宣称达到上限或显著突破。主分析保留既有不新增删样本的RF，补充版本用于敏感性及预测比较。

原QSH上仅增加训练尾部筛选的最小改动代码：`../code/RF_training_tail_sensitivity.py`，对应逐行差异在 `../diffs/RF_training_tail_sensitivity.diff`。该脚本已实跑，所得图表和因子组置换结果在RF_tail_sensitivity_results/；不作为主分析替换版。

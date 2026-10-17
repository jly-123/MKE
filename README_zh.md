# MKE

**Knowledge-Guided Multi-Granularity Subgraph Selection for GNN Explanations**

[English](README.md) · [方法定义](docs/METHOD.md) · [配置说明](docs/CONFIGURATION.md) · [实验指南](docs/EXPERIMENTS.md)

MKE 构建多粒度分子子结构候选，用 **Shapley group value** 评价候选组合，在节点并集预算下进行贪心选择。每一步比较加入候选后的组价值增量，选择可行候选中最大的正增益；完整保留选中的候选，重叠节点在预算中只计算一次。

仓库包含方法源码、图数据构建、GCN、训练与独立预测评估、六数据集配置、批量实验、结果汇总、绘图和测试。数据及权重按下面的目录另行准备。

## 文件说明

| 路径 | 用途 |
| --- | --- |
| `mke_new.py` | 候选生成、组价值计算、贪心选择和解释评估 |
| `train_gcn.py` | 预测器训练与检查点选择 |
| `evaluate_gcn.py` | 独立检查准确率、类别计数和混淆矩阵 |
| `mke_support/` | 图数据、GCN、池化、扰动评分及公共工具 |
| `config/` | 六数据集、模型及解释运行参数 |
| `scripts/check_environment.py` | 环境检查和可选的小型 CPU 前向检查 |
| `scripts/check_inputs.py` | CSV、标签、划分、权重文件及哈希检查 |
| `scripts/run_experiments.py` | 按数据集、稀疏度和种子批量运行，保存日志 |
| `scripts/summarize_results.py` | 从逐图记录复算指标，生成汇总表 |
| `scripts/plot_results.py` | 输出 PNG、JPG、PDF 指标图 |
| `scripts/render_explanations.py` | 在原分子上标出解释节点和内部化学键 |
| `test_mke_new_group_shapley.py`、`tests/` | 数学定义、适配器和实验工具测试 |
| `docs/`、`.github/` | 详细文档、自动测试和贡献模板 |

## 环境与输入

优先使用能够运行现有 GCN 的服务器环境。`requirements.txt` 是依赖清单，不是已验证的版本锁定文件。本仓库保留的 GCN 使用旧版 PyG 接口，安装最新版依赖不一定兼容。

```bash
python scripts/check_environment.py --smoke --output environment-report.json
python -m pip freeze > environment-server.txt
```

默认输入路径：

```text
data/origin_data/Mutagenicity.csv
checkpoints/Mutagenicity/gcn_3l_best.pth
```

CSV 需要 `smiles`、任务标签列及 `group`，划分值为 `training`、`valid`、`test`。检查点需包含兼容的 `net` 状态字典。

| 配置名称 | CSV 文件 | 标签列 | 隐藏维度 |
| --- | --- | --- | --- |
| `Mutagenicity` | `Mutagenicity.csv` | `Mutagenicity` | 256 |
| `bbbp` | `BBBP.csv` | `BBBP` | 128 |
| `hERG` | `hERG.csv` | `hERG` | 256 |
| `Tox21_AhR` | `Tox21_AhR.csv` | `AhR` | 256 |
| `Bace` | `Bace.csv` | `Bace` | 256 |
| `clintox` | `clintox.csv` | `CT_TOX` | 256 |

权重目录为 `checkpoints/<配置名称>/gcn_3l_best.pth`。保留 CSV 原划分，不重新随机切分；ClinTox 标签列须与检查点的训练任务对应。见[数据说明](docs/DATASETS.md)。

## 单次运行

```bash
python scripts/check_inputs.py --dataset Mutagenicity
python evaluate_gcn.py datasets=Mutagenicity device_id=0
python mke_new.py datasets=Mutagenicity device_id=0 target_sparsity=0.5
```

`device_id=0` 是当前进程可见的第一个 GPU。修改 `target_sparsity=0.6` 或 `0.7` 即可运行其他预算。

服务器已有输入时可以直接指定：

```bash
python mke_new.py datasets=Mutagenicity device_id=0 input_csv=/absolute/path/Mutagenicity.csv models.gnn_saving_dir=/absolute/path/checkpoints
```

节点预算为 `floor((1-target_sparsity)*num_nodes)`，实际稀疏度由最终选中节点数计算。完整候选、整数预算和正增益停止可能导致预算未用满，输出会分别记录目标值和实际值。

## 批量实验

先预览六数据集 × 三稀疏度的命令，再运行：

```bash
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --device 0
```

指定数据集、预算和采样种子：

```bash
python scripts/run_experiments.py --datasets Mutagenicity bbbp --sparsities 0.5 0.6 0.7 --seeds 0 1 2 --device 0
```

程序顺序运行，保存每个任务的命令、日志和状态，失败后停止后续任务。解释阶段的多个种子共享同一个指定预测器，改变的是置换采样；重新训练预测器属于单独流程。

## 汇总与画图

```bash
python scripts/summarize_results.py --results results --output reports
python scripts/plot_results.py --input reports/aggregated.csv --output figures --format png jpg pdf
```

从逐图概率复算 Fidelity±，从节点集和图大小复算实际稀疏度。不同数据、权重、标签及计算配置分别汇总。空解释仍计入均值，负值保留，先计算后舍入；单次运行不伪造跨运行标准差。

六数据集宏平均需要明确指定完整集合：

```bash
python scripts/summarize_results.py --results results --output reports --datasets Mutagenicity bbbp hERG Tox21_AhR Bace clintox
```

绘制分子解释：

```bash
python scripts/render_explanations.py --run /path/to/run_directory --samples 6904 6905 --output figures/explanations
```

请将示例编号替换为该次结果 CSV 中的 `sample_index`。默认生成 SVG，也支持 `--format png`。

## 训练与测试

训练新模型时使用新目录：

```bash
python train_gcn.py datasets=Mutagenicity device_id=0 models.gnn_saving_dir=checkpoints_new
```

源码和轻量测试不需要 Torch、PyG 或 RDKit：

```bash
python scripts/verify_repository.py
python test_mke_new_group_shapley.py
python -m unittest discover -s tests -v
```

完整 GNN、数据集及 GPU 检查需要在实际环境中执行。已完成的归档验证见 [VALIDATION.md](VALIDATION.md)。

[安装说明](docs/INSTALLATION.md) · [实验流程](docs/EXPERIMENTS.md) · [输出字段](docs/OUTPUTS.md) · [常见问题](docs/TROUBLESHOOTING.md) · [上传 GitHub](docs/GITHUB.md) · [引用](CITATION.md) · [许可](LICENSE) · [代码来源](THIRD_PARTY_NOTICES.md)

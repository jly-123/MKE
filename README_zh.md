# MKE

**Knowledge-Guided Multi-Granularity Subgraph Selection for GNN Explanations**

[English](README.md) · [方法](docs/METHOD.md) · [配置](docs/CONFIGURATION.md) · [实验指南](docs/EXPERIMENTS.md)

MKE 通过选择分子子结构解释已训练的图分类器：用 Shapley 组价值评价候选组合，在节点并集预算内贪心选择正增益最大的候选。

仓库包含训练与评估代码、六份实验 CSV 和六个历史 GCN 检查点。数据划分及已知问题见[数据说明](docs/DATASETS.md)。

## 目录

```text
data/origin_data/   分子数据 CSV
checkpoints/       按数据集分目录保存的 GCN 权重
config/            数据集、模型和解释器配置
mke_support/       数据加载、GCN、池化及评分
scripts/           批量实验、结果分析及绘图
tests/             工具测试
docs/              详细说明
mke_new.py         MKE 解释入口
train_gcn.py       预测器训练入口
evaluate_gcn.py    预测器评估入口
```

## 快速运行

在仓库根目录执行。GCN 使用旧版 PyG 接口，需要兼容的环境，见[安装说明](docs/INSTALLATION.md)。

```bash
python scripts/check_environment.py --smoke
python scripts/check_inputs.py --dataset Mutagenicity
python evaluate_gcn.py datasets=Mutagenicity device_id=0
python mke_new.py datasets=Mutagenicity device_id=0 target_sparsity=0.5
```

可选数据集为 `Mutagenicity`、`bbbp`、`hERG`、`Tox21_AhR`、`Bace` 和 `clintox`。节点预算为 `floor((1-target_sparsity)*num_nodes)`，结果另行记录实际稀疏度。

Tox21_AhR 和 ClinTox 存在跨划分重复，AhR 还存在冲突标签；实验前请查看[数据问题](docs/DATASETS.md#known-data-issues)及历史权重的任务来源说明。

## 训练与批量实验

```bash
python train_gcn.py datasets=Mutagenicity device_id=0 models.gnn_saving_dir=checkpoints_new
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --device 0
python scripts/summarize_results.py --results results --output reports
python scripts/plot_results.py --input reports/aggregated.csv --output figures
```

训练使用 Adam 和交叉熵，按验证集准确率选择最佳权重。新模型使用独立的权重目录。种子、外部路径和分子绘图见[实验指南](docs/EXPERIMENTS.md)。

## 源码检查

```bash
python scripts/verify_repository.py --check-manifest
python test_mke_new_group_shapley.py
python -m unittest discover -s tests -v
```

52 项轻量测试只需 Python 标准库；模型及 GPU 验证需要机器学习环境。

[输出格式](docs/OUTPUTS.md) · [常见问题](docs/TROUBLESHOOTING.md) · [GPLv3 许可](LICENSE) · [代码来源](THIRD_PARTY_NOTICES.md)

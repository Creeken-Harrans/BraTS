# Inference Guide

命令位置说明：

- 本文默认你当前目录就是 `BraTS` 项目根目录，所以命令写成 `python run.py ...`。
- 如果你当前在上一级 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径也都默认相对于 `BraTS` 项目根目录。

这份 guide 重点解释四件事：最佳配置选择、推理默认值、评估严格模式和报告链路的依赖关系。

## 调用链

### 最佳配置

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [find_best_configuration.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/find_best_configuration.py)

### 推理

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [predict_from_raw_data.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/predict_from_raw_data.py)

### 评估

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [evaluate_predictions.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/evaluate_predictions.py)

## 标准命令

```bash
python run.py train-all --npz
python run.py find-best-config
python run.py predict
python run.py evaluate --gt-dir ../nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations
python run.py report-evaluation
```

## `find-best-config` 真实在做什么

它会：

- 找出实际存在 validation 结果的候选模型；
- 只在 shared folds 上比较；
- 必要时尝试 ensemble；
- 搜索后处理；
- 生成正式推理说明文件。

如果候选模型之间没有任何共享 fold，它会直接失败，而不是输出不可比较的“最佳模型”。

## `predict` 依赖哪些东西

最少需要：

- 训练结果目录中的 checkpoint；
- 训练结果目录中的 `plans.json`；
- 训练结果目录中的 `dataset.json`；
- 合法命名的四模态输入。

如果你不显式传模型参数，CLI 会优先读取 `inference_information.json`。

## 默认输入命名契约

每个 case 必须包含：

- `{case_id}_0000.nii.gz`
- `{case_id}_0001.nii.gz`
- `{case_id}_0002.nii.gz`
- `{case_id}_0003.nii.gz`

对应语义：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

如果顺序错了，命令通常仍能执行，但结果语义会直接错。

## 自动抽样行为

默认输入目录 `04_inference_and_evaluation/input` 为空时，CLI 会：

- 从训练集抽样 `8` 个病例；
- 复制到 `input`；
- 写出 `input/sample_selection.json`。

如果你不希望这样，使用：

```bash
python run.py predict --disable-auto-sample-training
```

## `evaluate` 的真实限制

### 当前推荐写法

```bash
python run.py evaluate \
  --gt-dir ../nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations
```

### 为什么要显式传 `--gt-dir`

因为当前 CLI 的内置默认 `--gt-dir` 没有包含仓库外层所需的 `../`，与当前项目真实目录布局不一致。文档这里按真实可执行写法给出命令，代码问题已单独记录在 review findings 中。

### 子集评估

如果你评估的是自动抽样生成的子集，使用：

```bash
python run.py evaluate \
  --gt-dir ../nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations \
  --chill
```

## `report-evaluation` 依赖什么

默认依赖：

- `04_inference_and_evaluation/evaluation/summary.json`
- `04_inference_and_evaluation/predictions`
- `PROJECT_RAW/Dataset220_BraTS2020`

其中最后一项意味着当前 CLI 报告链路默认面向本项目数据集本身的病例。如果你评估的是完全外部的一批病例，报告 overlay 所需的原始模态目录需要单独处理。

## 相关文档

- [04_inference_and_evaluation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/README.md)
- [04_inference_and_evaluation/input/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/input/README.md)

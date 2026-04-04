# 03 Training And Results

命令位置说明：

- 本文默认你当前目录就是 `BraTS` 项目根目录，所以命令写成 `python run.py ...`。
- 如果你当前在上一级 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径也都默认相对于 `BraTS` 项目根目录。

这一阶段负责把 preprocess 后的数据真正送入训练器，并把 checkpoint、validation 结果、日志和后续推理所需的模型产物写到固定位置。

## 默认训练约定

当前真实默认值来自 [project_config.json](/home/Creeken/Desktop/machine-learning-test/BraTS/project_config.json) 和 [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)：

- trainer: `SegTrainer`
- plans: `ProjectPlans`
- configuration: `3d_fullres`
- epochs: `150`
- folds: `0 1 2 3 4`

## 训练命令

### 先验证单 fold

```bash
python run.py train --fold 0
```

这条命令适合先确认：

- preprocess 输出是否完整；
- trainer / plans / configuration 名字是否匹配；
- 结果目录是否可写；
- GPU 或 CPU 设备是否可用；
- checkpoint 恢复逻辑是否正常。

### 完整五折

```bash
python run.py train-all --npz
```

推荐带 `--npz` 的原因：

- `find-best-config` 需要完整 validation 结果；
- ensemble 依赖 validation 概率输出；
- 当前项目的标准主线就是 `train-all --npz -> find-best-config -> predict -> evaluate`。

### 只做验证

```bash
python run.py train --fold 0 --validation-only --npz
```

## 输入、输出和目录语义

### 训练主要读取

- `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/dataset.json`
- `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans.json`
- `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/splits_final.json`
- 对应 configuration 的 preprocess 输出目录

### 训练主要写入

- `03_training_and_results/artifacts/nnUNet_results/Dataset220_BraTS2020/<trainer>__<plans>__<configuration>/fold_x`

其中最关键的训练产物包括：

- `checkpoint_latest.pth`
- `checkpoint_best.pth`
- `checkpoint_final.pth`
- `training_log_*.txt`
- `progress.png`
- `validation/*.npz`
- `validation/summary.json`

注意：

- `validation/*.nii.gz` 会被自动外移到 `../BraTS-Dataset/workspace_nifti/...`
- 工作目录中的 `validation/*.nii.gz` 会保留为兼容 symlink，方便原有读取逻辑继续工作

### 仓库内同步快照

训练阶段会把部分结果同步到仓库内，方便阅读和交付：

- `03_training_and_results/results/fold_x`

这里是快照，不是训练的唯一真实结果目录。真正的训练主产物现在在 `03_training_and_results/artifacts/nnUNet_results/...`。

## 恢复训练的真实行为

当前 CLI 在进入 trainer 前会先检查已有 checkpoint。

自动恢复优先级：

1. `checkpoint_final.pth`
2. `checkpoint_latest.pth`
3. `checkpoint_best.pth`

意味着：

- 如果当前 fold 已经有可恢复 checkpoint，`python run.py train --fold 0` 会自动走续训路径。
- 如果没有可恢复 checkpoint，命令会从头开始。
- 如果 checkpoint 是旧的总轮数，例如已经完成 `100` 轮，而当前 `SegTrainer` 已经改成 `150` 轮，命令会继续从已有 checkpoint 接着训练到 `150`。
- 如果你明确要忽略已有结果并从头开始，使用 `--restart-training`。

## `fold` 的含义

`fold` 不是“第几次训练”，而是 `splits_final.json` 中的一组 train/validation 划分。

因此：

- `fold_0` 是第 0 组交叉验证划分；
- `train-all` 就是按 `fold_0 -> fold_1 -> fold_2 -> fold_3 -> fold_4` 的顺序遍历默认五折；
- 先跑 `fold 0` 的意义是先验证工程链路，不是只做一次随手实验。

## 训练前后最该检查什么

### 训练前

- `plan-preprocess` 是否已经完成；
- `ProjectPlans.json` 和 `dataset.json` 是否存在；
- 目标 configuration 的预处理目录是否完整；
- `doctor` 是否能正确识别项目路径和数据目录。

### 训练后

- `training_log_*.txt` 是否明确记录了本次是新训练还是从哪个 checkpoint 恢复；
- `progress.png` 是否持续更新；
- `validation/summary.json` 是否已经生成；
- 完整五折后是否已经具备 `find-best-config` 所需的 shared folds。

## 哪些文件可删，哪些不要乱删

### 可以重建

- `03_training_and_results/results/*`
  仓库内快照，可以重新同步生成。

### 不要随便删

- `03_training_and_results/artifacts/nnUNet_results/Dataset220_BraTS2020/...`
  真正的 checkpoint、validation 概率和 cross-validation 资产。
- `../BraTS-Dataset/workspace_nifti/...`
  训练验证阶段被自动外移的历史 NIfTI 真正存放位置。

## 相关代码入口

- [run_training.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/run/run_training.py)
- [SegTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/SegTrainer.py)
- [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)
- [training_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/docs/training_guide.md)
- [code_directory_audit.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/docs/code_directory_audit.md)

# Training Guide

命令位置说明：

- 本文默认你当前目录就是 `BraTS` 项目根目录，所以命令写成 `python run.py ...`。
- 如果你当前在上一级 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径也都默认相对于 `BraTS` 项目根目录。

这份 guide 只讲训练主链最容易出问题的部分：入口、依赖、恢复逻辑、结果目录和排查顺序。

## 调用链

训练主链按顺序是：

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [run_training.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/run/run_training.py)
4. [SegTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/SegTrainer.py)
5. [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)

## 标准训练命令

```bash
python run.py train --fold 0
python run.py train-all --npz
python run.py train --fold 0 --validation-only --npz
```

当前默认训练目标：

- trainer: `SegTrainer`
- configuration: `3d_fullres`
- plans: `ProjectPlans`
- epochs: `150`
- folds: `0 1 2 3 4`

## trainer 启动前必须满足的前提

至少要有：

- `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/dataset.json`
- `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans.json`
- `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/splits_final.json`
- 当前 configuration 对应的 preprocess 输出目录

如果这些不完整，CLI 会在进入 trainer 前直接失败，而不是等训练循环里再报模糊错误。

## 恢复逻辑

当前真实行为：

- `train` 会先检查已有 checkpoint；
- 检查优先级：`checkpoint_final.pth -> checkpoint_latest.pth -> checkpoint_best.pth`；
- 找到后自动进入恢复路径；
- 如果 checkpoint 对应的是更早的总轮数，例如 `100`，而当前 trainer 的目标总轮数已经提高到 `150`，则会在已有 checkpoint 基础上继续训练到新的目标轮数；
- 没找到时从头开始；
- 只有显式传 `--restart-training` 才会忽略已有 checkpoint。

因此，`python run.py train --fold 0` 在当前项目里更接近“把 fold_0 放到一个可继续执行的状态”，而不是保证每次都从 epoch 0 开始。

## 为什么建议先跑 `fold 0`

因为最先需要确认的是工程链路，而不是五折最终分数：

- trainer 是否能被正确找到；
- preprocess 目录是否完整；
- 输出目录是否可写；
- checkpoint 是否能正确恢复；
- 设备选择是否正确。

## 为什么推荐 `train-all --npz`

因为当前下游主线依赖 validation 概率输出：

- `find-best-config`
- ensemble
- 正式推理方案选择

如果你只做单 fold 调试，`--npz` 不是必需；如果你要走完整流程，建议保留。

另外，`train-all` 的执行顺序是固定的：

- `fold_0 -> fold_1 -> fold_2 -> fold_3 -> fold_4`

如果你重启 `train-all`，它会从 `fold_0` 先开始检查是续训还是跳过，而不是直接跳到当前最后一个 fold。

## 结果目录怎么读

重点看下面几类文件：

- `training_log_*.txt`
  看本次到底是新训练还是恢复训练，以及恢复来源。
- `progress.png`
  看训练和验证曲线是否正常推进。
- `checkpoint_latest.pth`
  每轮更新的恢复点。
- `checkpoint_best.pth`
  当前最佳权重。
- `checkpoint_final.pth`
  最终保存的收尾权重。
- `validation/summary.json`
  held-out fold 的最终 validation 结果。

## 常见排查顺序

1. 先跑 `python run.py doctor`
2. 再跑 `python run.py plan-preprocess`
3. 再跑 `python run.py train --fold 0`
4. 最后再执行 `python run.py train-all --npz`

## 相关文档

- [03_training_and_results/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/README.md)
- [04_inference_and_evaluation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/README.md)

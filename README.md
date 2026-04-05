# BraTS2020 Project

这个仓库是一个围绕 BraTS2020 数据集整理的本地 Python 工程。它把数据准备、preprocess、训练、推理、评估和报告统一到一个入口 `run.py` 下，并且把每个阶段的输入、输出和说明文件放在固定位置。

## 快速开始

### 先分清命令写法

同一个仓库有两种合法调用方式，只取决于你当前站在哪个目录：

| 你当前所在目录 | 正确命令写法 |
| --- | --- |
| `.../machine-learning-test/BraTS` | `python run.py ...` |
| `.../machine-learning-test` | `python BraTS/run.py ...` |

本文后续命令默认都按第一种写法展示，也就是默认你已经进入 `BraTS` 仓库根目录。

### 先验证入口

```bash
python run.py --help
python run.py doctor
```

如果你在上一级目录执行，对应命令是：

```bash
python BraTS/run.py --help
python BraTS/run.py doctor
```

### 标准主线命令

```bash
python run.py prepare-dataset
python run.py plan-preprocess
python run.py train --fold 0
python run.py train-all --npz
python run.py find-best-config
python run.py predict
python run.py evaluate
python run.py report-evaluation
```

说明：

- `train --fold 0` 适合先验证训练链路。
- `train-all --npz` 是完整五折训练的标准命令。
- `evaluate` 现在可以直接使用默认 `--gt-dir`，它会指向 `BraTS-Dataset/nnUNet_preprocessed/.../gt_segmentations`。

## 仓库结构

### 根目录内主要内容

- `run.py`
  统一 CLI 入口。真正调用的是 `03_training_and_results/src/brats_project/cli.py`。
- `doctor.py`
  入口包装脚本，等价于 `python run.py doctor`。
- `project_config.json`
  本项目默认数据集、路径和运行时参数的集中配置。
- `00_first_case_visualization/`
  原始病例可视化和数据直觉建立。
- `01_data_preparation/`
  原始 BraTS 病例转换成 `Dataset220_BraTS2020`。
- `02_preprocess/`
  fingerprint、plans、preprocess、日志和元数据快照。
- `03_training_and_results/`
  训练主代码和训练结果样例快照。
- `04_inference_and_evaluation/`
  推理输入、预测输出、评估结果和报告输出。
- `tests/`
  冒烟与回归测试。

### 仓库外但由本项目默认使用的目录

这些路径来自 [project_config.json](/home/Creeken/Desktop/machine-learning-test/BraTS/project_config.json)：

- `../BraTS-Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`
  原始 BraTS 数据根目录。
- `../BraTS-Dataset/nnUNet_raw`
  转换后的 raw 数据目录。
- `../BraTS-Dataset/nnUNet_preprocessed`
  fingerprint、plans、preprocess 和 `gt_segmentations`。
- `../BraTS-Dataset/inference`
  推理输入和推理输出目录，所有 `.nii` / `.nii.gz` 都应该落在这里。
- `03_training_and_results/artifacts/nnUNet_results`
  checkpoint、validation 概率、cross-validation 汇总和 `inference_information.json`。
- `../BraTS-Dataset/workspace_nifti`
  从工作目录自动外移的 NIfTI 镜像目录。训练验证产生的 `.nii.gz` 会物理落在这里，工作目录保留兼容 symlink。

## 路径规则

这一点必须明确，因为它直接决定命令是否能跑通：

- CLI 里的相对路径不是相对于你当前 shell 的 `cwd` 解释，而是统一相对于 `BraTS` 项目根目录解释。
- 因此，即使你在上一级目录执行 `python BraTS/run.py ...`，CLI 参数里的相对路径仍然要按 `BraTS` 根目录来写。
- 例如 `predict` 的默认输入目录现在是 `../BraTS-Dataset/inference/input`，不是 `BraTS/04_inference_and_evaluation/input`。
- 对于仓库外目录，要显式写 `../...`，例如：
  - `../BraTS-Dataset/archive/...`
  - `../BraTS-Dataset/nnUNet_raw`
  - `../BraTS-Dataset/nnUNet_preprocessed/...`
  - `../BraTS-Dataset/inference/...`

### 典型对照

| 目的 | 写法 |
| --- | --- |
| 推理输入目录 | `../BraTS-Dataset/inference/input` |
| 推理输出目录 | `../BraTS-Dataset/inference/predictions` |
| ground truth 目录 | `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations` |
| 原始 BraTS 数据目录 | `../BraTS-Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData` |
| 训练结果目录 | `03_training_and_results/artifacts/nnUNet_results` |

## 命令清单

### 环境与入口

- `python run.py --help`
  查看 CLI 总帮助，并确认当前应该使用 `python run.py ...` 还是 `python BraTS/run.py ...`。
- `python run.py doctor`
  打印项目根目录、默认数据集和关键路径状态。
- `python doctor.py`
  `doctor` 的包装入口。

### 数据准备

- `python run.py visualize-first-case`
  读取原始 BraTS 数据，生成首例病例可视化到 `00_first_case_visualization/output`。
- `python run.py prepare-dataset`
  将原始 BraTS 病例转换成 `../BraTS-Dataset/nnUNet_raw/Dataset220_BraTS2020`。
- `python run.py prepare-dataset --force`
  强制重建上述 raw 数据目录。

### Preprocess

- `python run.py plan-preprocess`
  复用已有 fingerprint、plans 和 preprocess 产物；缺什么补什么。
- `python run.py plan-preprocess --verify-dataset --configurations 3d_fullres`
  在 preprocess 前做数据完整性检查，并只处理 `3d_fullres`。
- `python run.py plan-preprocess --recompute-fingerprint`
  只重算 `dataset_fingerprint.json`。
- `python run.py plan-preprocess --recompute-plans`
  只重算 `ProjectPlans.json`。
- `python run.py plan-preprocess --force-preprocess`
  只重做预处理输出。
- `python run.py plan-preprocess --clean`
  三者都重做。

### 训练

- `python run.py train --fold 0`
  先跑一个 fold 验证训练链路。
- `python run.py train --fold 0 --validation-only --npz`
  只对现有 fold 结果做最终 validation。
- `python run.py train-all --npz`
  跑默认五折，并保留 validation 概率输出，供 `find-best-config` 和 ensemble 使用。

训练默认值与真实实现一致：

- trainer: `SegTrainer`
- plans: `ProjectPlans`
- configuration: `3d_fullres`
- epochs: `150`
- folds: `0 1 2 3 4`

训练恢复规则与真实实现一致：

- 如果检测到当前 fold 已存在 checkpoint，CLI 会自动进入续训路径。
- 自动恢复优先级是 `checkpoint_final.pth -> checkpoint_latest.pth -> checkpoint_best.pth`。
- 如果当前 checkpoint 对应的是较早的总轮数，而现在 trainer 的目标总轮数更高，例如从 `100` 提到 `150`，CLI 会继续从已有 checkpoint 续训到新的目标轮数。
- 恢复时以 checkpoint 内部保存的 epoch 和 logger 状态为准；`training_state.json` 或旧 `training_log_*.txt` 只用于辅助诊断，不会再把恢复进度推进到 checkpoint 之外。
- 如果历史文本日志里出现比 checkpoint 更靠后的 `next_epoch`，训练会打印 warning 并忽略这类元数据，避免出现恢复到不存在 epoch 后的 logger 越界。
- `train-all` 会按 `fold_0 -> fold_1 -> fold_2 -> fold_3 -> fold_4` 的顺序逐个检查并执行续训或新训练。
- 如果你要强制从头开始，显式传 `--restart-training`。

### 推理、评估与报告

- `python run.py find-best-config`
  汇总 shared folds 上的 validation 结果，确定最佳单模型或 ensemble，并写出 `inference_information.json`。
- `python run.py predict`
  默认从 `../BraTS-Dataset/inference/input` 读取，输出到 `../BraTS-Dataset/inference/predictions`。
- `python run.py predict --sample-training-cases 12 --sample-seed 123`
  从训练集随机抽样病例填充输入目录后执行推理。
- `python run.py predict --disable-auto-sample-training`
  禁止在默认输入目录为空时自动抽样。
- `python run.py evaluate`
  评估预测目录与 GT 目录。
- `python run.py evaluate --chill`
  允许只评估一个预测子集。自动抽样出的临时验证集通常需要这一写法。
- `python run.py report-evaluation`
  从 `04_inference_and_evaluation/evaluation/summary.json` 和 `../BraTS-Dataset/inference/predictions` 生成 Markdown 报告和可视化。

## 输入、输出和结果语义

### 训练相关

- 输入：
  - `../BraTS-Dataset/nnUNet_raw/Dataset220_BraTS2020`
  - `../BraTS-Dataset/nnUNet_preprocessed/Dataset220_BraTS2020`
- 输出：
  - `03_training_and_results/artifacts/nnUNet_results/Dataset220_BraTS2020/...`
- 仓库内同步快照：
  - `02_preprocess/metadata/*`
  - `02_preprocess/logs/*`
  - `03_training_and_results/results/fold_*/*`

### 推理相关

- 默认输入目录：`../BraTS-Dataset/inference/input`
- 默认预测目录：`../BraTS-Dataset/inference/predictions`
- 默认评估输出：`04_inference_and_evaluation/evaluation/summary.json`
- 默认报告输出：`04_inference_and_evaluation/report`

### `input / predictions / evaluation / report` 的含义

- `input`
  推理输入目录。里面应该放待推理病例的四模态 NIfTI，或自动抽样生成的临时输入。
- `predictions`
  推理输出目录。只放 `.nii.gz` 预测分割。推理元数据会转存到 `04_inference_and_evaluation/metadata`。
- `evaluation`
  评估结果目录。默认核心文件是 `summary.json`。
- `report`
  基于 `summary.json` 进一步生成的 Markdown 报告、图表、病例分析和 overlay 图。

## 重新生成哪些结果

### 重新生成 `input`

- 手动删除 `../BraTS-Dataset/inference/input` 里的病例文件后重新放入新病例。
- 如果你依赖自动抽样，也可以删除其中的 `.nii.gz`，并删除 `04_inference_and_evaluation/metadata/sample_selection.json`，再重新执行 `python run.py predict`。

### 重新生成 `predictions`

- 删除 `../BraTS-Dataset/inference/predictions` 下的预测输出后重新执行 `python run.py predict`。

### 重新生成 `evaluation`

- 删除 `04_inference_and_evaluation/evaluation/summary.json` 后重新执行 `python run.py evaluate`。

### 重新生成 `report`

- 删除 `04_inference_and_evaluation/report` 后重新执行 `python run.py report-evaluation`。

## 哪些文件可以删，哪些不要删

### 可以安全重建的仓库内产物

- `00_first_case_visualization/output`
- `02_preprocess/logs`
- `02_preprocess/metadata`
- `03_training_and_results/results`
- `04_inference_and_evaluation/metadata`
- `04_inference_and_evaluation/evaluation`
- `04_inference_and_evaluation/report`

### 需要谨慎处理的目录

- `../BraTS-Dataset/inference/input`
  里面可能是你手工放入的真实推理输入，不要在不了解内容时整目录删除。
- `../BraTS-Dataset/nnUNet_raw`
  这是训练数据集转换结果，删掉后需要重新 `prepare-dataset`。
- `../BraTS-Dataset/nnUNet_preprocessed`
  这是 preprocess 产物和 `gt_segmentations`，删掉后需要重新 `plan-preprocess`。
- `03_training_and_results/artifacts/nnUNet_results`
  这是 checkpoint、validation 概率和 `find-best-config` 依赖的主结果目录。
- `../BraTS-Dataset/workspace_nifti`
  这是从工作目录外移出去的历史/运行期 NIfTI 真正存放位置，删掉会破坏工作目录里的兼容 symlink。

## 常见错误

### 错误 1：在 `BraTS` 根目录里执行 `python BraTS/run.py ...`

现象：

```text
python: can't open file '/.../BraTS/BraTS/run.py': [Errno 2] No such file or directory
```

原因：

- 你已经在 `BraTS` 根目录里，不应该再写 `BraTS/run.py`。

正确写法：

- 在 `BraTS` 根目录：`python run.py ...`
- 在上一级目录：`python BraTS/run.py ...`

### 错误 2：显式路径忘了写 `../`

例如原始数据和 `BraTS-Dataset` 都在仓库外层，所以显式路径应该写成：

- `../BraTS-Dataset/archive/...`
- `../BraTS-Dataset/nnUNet_raw/...`
- `../BraTS-Dataset/nnUNet_preprocessed/...`
- `../BraTS-Dataset/inference/...`

不要写成：

- `archive/...`
- `BraTS-Dataset/...`

除非该路径本来就位于 `BraTS` 仓库内部。

### 错误 3：`predict` 能跑，但结果语义不对

通常是输入命名契约错误：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

### 错误 4：`evaluate` 提示覆盖不一致

原因通常是：

- 你评估的是预测子集；
- 自动抽样只预测了一部分训练病例；
- 或预测目录与 GT 目录文件名不完全对应。

如果你是有意评估一个子集，显式传 `--chill`。

## 验证与测试

当前仓库至少应验证：

```bash
python run.py --help
python run.py doctor
python BraTS/run.py --help
python BraTS/run.py doctor
python -m unittest tests.test_smoke tests.test_regressions
```

## 进一步阅读

- [00_first_case_visualization/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/00_first_case_visualization/README.md)
- [01_data_preparation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/README.md)
- [01_data_preparation/docs/data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/docs/data_contract.md)
- [02_preprocess/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/README.md)
- [02_preprocess/docs/preprocess_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/docs/preprocess_guide.md)
- [03_training_and_results/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/README.md)
- [03_training_and_results/docs/training_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/docs/training_guide.md)
- [04_inference_and_evaluation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/README.md)
- [04_inference_and_evaluation/docs/inference_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/docs/inference_guide.md)
- [PIPELINE_EXPLANATION.md](/home/Creeken/Desktop/machine-learning-test/BraTS/PIPELINE_EXPLANATION.md)
- [explain.md](/home/Creeken/Desktop/machine-learning-test/BraTS/explain.md)

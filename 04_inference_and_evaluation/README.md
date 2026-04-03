# 04 Inference And Evaluation

命令位置说明：

- 本文默认你当前目录就是 `BraTS` 项目根目录，所以命令写成 `python run.py ...`。
- 如果你当前在上一级 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径也都默认相对于 `BraTS` 项目根目录。

这一阶段负责把训练产物变成最终可读的预测、评估结果和报告。

## 标准顺序

```bash
python run.py train-all --npz
python run.py find-best-config
python run.py predict
python run.py evaluate --gt-dir ../nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations
python run.py report-evaluation
```

说明：

- `find-best-config` 会基于 shared folds 比较候选模型；
- `predict` 默认从 `04_inference_and_evaluation/input` 读取；
- `evaluate` 当前请显式传 `--gt-dir ../nnUNet_test/.../gt_segmentations`，因为 CLI 内置默认 `--gt-dir` 与当前仓库路径解析规则不一致；
- `report-evaluation` 读取 `evaluation/summary.json` 和 `predictions/`，生成图表与病例分析。

## 目录说明

### `input`

- 目录：`04_inference_and_evaluation/input`
- 作用：推理输入目录。
- 内容：每个 case 需要四个模态文件：
  - `{case_id}_0000.nii.gz`
  - `{case_id}_0001.nii.gz`
  - `{case_id}_0002.nii.gz`
  - `{case_id}_0003.nii.gz`

### `predictions`

- 目录：`04_inference_and_evaluation/predictions`
- 作用：推理输出目录。
- 典型内容：
  - `*.nii.gz`
  - `dataset.json`
  - `plans.json`
  - `predict_from_raw_data_args.json`

### `evaluation`

- 目录：`04_inference_and_evaluation/evaluation`
- 作用：评估结果目录。
- 核心文件：`summary.json`

### `report`

- 目录：`04_inference_and_evaluation/report`
- 作用：报告输出目录。
- 典型内容：
  - `report.md`
  - `analysis.json`
  - `case_metrics.csv`
  - `case_analysis.csv`
  - `*.png`
  - `cases/`

## `find-best-config` 在做什么

它不是一个装饰性汇总命令，而是正式推理前的决策步骤。

它会：

- 找出当前真实存在 validation 结果的模型；
- 只在所有候选模型共同拥有的 folds 上比较；
- 如允许，尝试 ensemble；
- 决定后处理；
- 写出：
  - `../nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_information.json`
  - `../nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_instructions.txt`

## `predict` 的真实默认行为

### 默认输入输出

- 输入：`04_inference_and_evaluation/input`
- 输出：`04_inference_and_evaluation/predictions`

### 自动抽样

如果默认输入目录为空，且没有显式禁用自动抽样，CLI 会：

- 从 `PROJECT_RAW/Dataset220_BraTS2020/imagesTr` 随机抽样 `8` 个训练病例；
- 复制到 `04_inference_and_evaluation/input`；
- 把抽样记录写到 `04_inference_and_evaluation/input/sample_selection.json`。

可选写法：

```bash
python run.py predict --sample-training-cases 12 --sample-seed 123
python run.py predict --disable-auto-sample-training
```

### 模型参数默认来源

如果你没有显式传 `--trainer/--configuration/--plans/--folds`，`predict` 会优先读取 `find-best-config` 生成的 `inference_information.json`。

## `evaluate` 的真实行为

### 标准命令

```bash
python run.py evaluate --gt-dir ../nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations
```

### 子集评估

如果你评估的是预测子集，例如自动抽样出的临时验证病例，使用：

```bash
python run.py evaluate \
  --gt-dir ../nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations \
  --chill
```

### 严格模式

当前默认行为是严格模式：

- 预测目录为空时直接报错；
- 预测文件名与 GT 文件名不一致时直接报错；
- 只有显式传 `--chill` 才允许评估子集。

## `report-evaluation` 的真实输入输出

默认读取：

- `04_inference_and_evaluation/evaluation/summary.json`
- `04_inference_and_evaluation/predictions`
- 如果存在，还会读取 `04_inference_and_evaluation/input/sample_selection.json`

默认写入：

- `04_inference_and_evaluation/report`

说明：

- 入口层已修复 `report-evaluation` 的默认 `sample_selection_file` 和 `output_dir` 路径前缀，不再错误地附带 `BraTS/`。
- 当前 CLI 会把 `raw_dataset_dir` 固定到 `PROJECT_RAW/<dataset>`。这意味着报告的 overlay 能稳定支持项目默认训练数据集，但如果你评估的是完全外部的一套病例，报告链路需要单独提供匹配的原始模态目录。这个限制已在代码审查中记录。

## 哪些内容可以重新生成

- 删除 `predictions/` 后重新执行 `python run.py predict`
- 删除 `evaluation/summary.json` 后重新执行 `python run.py evaluate --gt-dir ../nnUNet_test/.../gt_segmentations`
- 删除 `report/` 后重新执行 `python run.py report-evaluation`

## 相关文档

- [input/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/input/README.md)
- [docs/inference_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/docs/inference_guide.md)
- [find_best_configuration.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/find_best_configuration.py)
- [predict_from_raw_data.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/predict_from_raw_data.py)
- [evaluate_predictions.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/evaluate_predictions.py)

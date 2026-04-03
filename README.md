# BraTS2020 Project

这个目录不是“几个训练脚本的合集”，而是一套围绕你当前工作区、当前 BraTS2020 数据、当前 nnU-Net 派生代码整理出来的完整项目。

它有三个非常明确的目标：

1. 把原始 BraTS 数据转换成当前项目能稳定训练的格式。
2. 把 nnU-Net 风格的自动规划、预处理、训练、推理链路本地化成统一入口。
3. 把“数据 -> plans -> 训练 -> 推理 -> 评估”的每一层都留下可读文档和可检查产物。

如果只用一句话概括整个项目：

- 它是一个“本地化、流程化、文档化”的 BraTS2020 分割工程，而不是上游 nnU-Net 命令的简单搬运。

---

## 1. 你先应该建立什么整体认识

这个项目的主线不是“先看网络结构”，而是下面这条工程链：

1. 看懂原始 BraTS 多模态 3D MRI 和标签
2. 把原始病例整理成 `Dataset220_BraTS2020`
3. 提取 fingerprint，生成 plans，完成 preprocess
4. 用 `SegTrainer + ProjectPlans + 3d_fullres` 跑训练
5. 汇总交叉验证结果，找最佳配置，做正式推理与评估

你真正要先搞清楚的事情是：

- 输入数据在磁盘上怎样组织
- `dataset.json` 怎样定义任务
- `dataset_fingerprint.json` 在记录什么
- `ProjectPlans.json` 为什么长成现在这样
- 训练结果会写到哪里
- 推理和后处理到底依赖哪些训练产物

---

## 2. 顶层目录说明

### `00_first_case_visualization/`

作用：建立 BraTS 原始病例的空间直觉、模态直觉、标签直觉。

这一层的价值不是“画几张图”，而是先回答后面所有代码都默认成立的问题：

- 四个模态和一个标签是不是严格对齐
- 标签值到底表示什么
- 3D volume 在三个平面上长什么样
- 不同模态分别更容易看见什么病灶信息

### `01_data_preparation/`

作用：把原始 BraTS 目录转换成本项目训练目录。

这一层会把原始病例重组为：

- `imagesTr/`
- `labelsTr/`
- `dataset.json`

并且统一：

- 通道顺序
- 文件命名
- 标签映射
- region-based training 所需的任务定义

### `02_preprocess/`

作用：把“训练目录”进一步变成“训练器真正读取的数据和计划”。

这一层做四件事：

1. 提取 dataset fingerprint
2. 根据 fingerprint 生成 experiment plans
3. 执行 preprocess
4. 准备交叉验证 split

### `03_training_and_results/`

作用：放置训练核心代码和训练结果样例。

这里最重要的部分是：

- `src/brats_project/`
  - 核心 Python 包
- `results/`
  - 历史训练输出样例，用于帮助理解产物结构

### `04_inference_and_evaluation/`

作用：承接训练之后的正式推理和评估阶段。

这里包含：

- 推理输入目录
- 推理输出目录
- 推理/评估文档

### `Document/`

作用：把项目 Markdown 文档批量导出成 PDF。

它不参与模型训练，但参与项目文档交付。

---

## 3. 统一入口与配置

### 入口

项目统一入口是：

- [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)

它本身不做训练，只做两件事：

1. 把 `03_training_and_results/src` 放进 Python 搜索路径
2. 把控制权交给 `brats_project.cli`

所以真正的控制台核心在：

- [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)

### 配置

项目的默认配置集中在：

- [project_config.json](/home/Creeken/Desktop/machine-learning-test/BraTS/project_config.json)

它管理三类默认值：

- 数据集默认值
  - `id=220`
  - `name=Dataset220_BraTS2020`
  - `trainer=SegTrainer`
  - `plans=ProjectPlans`
  - `default_configuration=3d_fullres`
- 路径默认值
  - 原始 BraTS 根目录
  - raw/preprocessed/results 根目录
  - 可视化输出目录
  - 推理输入输出目录
- 运行环境默认值
  - 期望的 PyTorch/CUDA
  - 默认并发参数

你可以把它理解成“项目约定的总开关”，而不是“算法配置文件”。

### 路径解析

路径相关代码主要在：

- [project_layout.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/project_layout.py)
- [paths.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/paths.py)

它们负责：

- 找项目根目录
- 找工作区根目录
- 把相对路径解释成工作区路径
- 设置并导出 `PROJECT_RAW / PROJECT_PREPROCESSED / PROJECT_RESULTS`

---

## 4. 真实执行顺序

在真正跑命令前，先把最关键的“文件从哪里来，到哪里去”看清楚。

### 总数据流

1. 原始 BraTS 训练集：
   `archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`
2. 转换后的项目训练数据：
   `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020`
3. fingerprint、plans、预处理结果：
   `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020`
4. checkpoint、validation、crossval 汇总、最佳配置信息：
   `nnUNet_test/nnUNet_results/Dataset220_BraTS2020`
5. 项目内默认推理输入：
   `BraTS/04_inference_and_evaluation/input`
6. 项目内默认推理输出：
   `BraTS/04_inference_and_evaluation/predictions`

### 每一步到底读什么，写什么

如果你关心的是“我现在敲这一条命令，它到底从哪里读、往哪里写”，直接看下面这个清单。

#### `python BraTS/run.py doctor`

读取：

- `BraTS/project_config.json`
- 环境变量 `PROJECT_RAW / PROJECT_PREPROCESSED / PROJECT_RESULTS`
- `torch` / CUDA 运行时

写入：

- 不写文件，只向终端打印环境检查结果

#### `python BraTS/run.py visualize-first-case`

读取：

- `archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

写入：

- `BraTS/00_first_case_visualization/output`

#### `python BraTS/run.py prepare-dataset`

读取：

- `archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

写入：

- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/imagesTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/labelsTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/dataset.json`
- `BraTS/01_data_preparation/metadata/raw_dataset.json`
- `BraTS/02_preprocess/metadata/dataset.json`

#### `python BraTS/run.py plan-preprocess`

读取：

- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/imagesTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/labelsTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/dataset.json`

写入：

- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/dataset_fingerprint.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans_2d/*`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans_3d_fullres/*`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/splits_final.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations/*`
- `BraTS/02_preprocess/metadata/dataset_fingerprint.json`
- `BraTS/02_preprocess/metadata/ProjectPlans.json`
- `BraTS/02_preprocess/metadata/splits_final.json`
- `BraTS/02_preprocess/logs/preprocess_*.log`
- `BraTS/02_preprocess/logs/latest.log`

#### `python BraTS/run.py train --fold 0`

读取：

- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/dataset.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans_3d_fullres/*`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/splits_final.json`
- 如果已有断点续训：
  `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0/checkpoint_*.pth`

写入：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0/checkpoint_best.pth`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0/checkpoint_final.pth`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0/checkpoint_latest.pth`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0/training_log_*.txt`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0/progress.png`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0/validation/*`
- `BraTS/03_training_and_results/results/fold_0/*`

#### `python BraTS/run.py train-all --npz`

读取：

- 和 `train --fold 0` 相同，只是对 `fold_0 ... fold_4` 逐个读取

写入：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0 ... fold_4/*`
- 每个 `fold_x/validation/` 里额外写出 `.npz` 概率文件

#### `python BraTS/run.py find-best-config`

读取：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/*/fold_x/validation/*`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/*/fold_x/validation/*.npz`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations/*`

写入：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<model>/crossval_results_folds_*/summary.json`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<model>/crossval_results_folds_*/postprocessing.pkl`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_information.json`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_instructions.txt`

#### `python BraTS/run.py predict`

读取模型：

- 默认优先读
  `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_information.json`
- 再去读对应模型目录：
  `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__<plans>__<configuration>/fold_x/checkpoint_*.pth`

读取输入：

- 默认读 `BraTS/04_inference_and_evaluation/input/*.nii.gz`
- 如果这里没有输入病例，默认会从
  `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/imagesTr/*.nii.gz`
  随机抽样一批病例复制进来

写入：

- `BraTS/04_inference_and_evaluation/input/sample_selection.json`
- `BraTS/04_inference_and_evaluation/predictions/*.nii.gz`
- `BraTS/04_inference_and_evaluation/predictions/dataset.json`
- `BraTS/04_inference_and_evaluation/predictions/plans.json`
- `BraTS/04_inference_and_evaluation/predictions/predict_from_raw_data_args.json`

#### `python BraTS/run.py evaluate`

读取：

- `BraTS/04_inference_and_evaluation/predictions/*.nii.gz`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations/*.nii.gz`
- 优先读
  `BraTS/04_inference_and_evaluation/predictions/dataset.json`
- 优先读
  `BraTS/04_inference_and_evaluation/predictions/plans.json`
- 如果预测目录里没有元数据，则回退读取
  `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__<plans>__<configuration>/dataset.json`
- 如果预测目录里没有元数据，则回退读取
  `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__<plans>__<configuration>/plans.json`

写入：

- 默认写到 `BraTS/04_inference_and_evaluation/evaluation/summary.json`
- 如果指定 `--output-file`，就写到该路径

#### `visualize-first-case`

从哪里读：

- `archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

写到哪里：

- `BraTS/00_first_case_visualization/output`

产物是什么：

- 首个病例的多模态切片图
- 标签可视化图
- 帮助理解数据结构的说明文件

#### `prepare-dataset`

从哪里读：

- `archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

写到哪里：

- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/imagesTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/labelsTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/dataset.json`

顺手同步到哪里：

- `BraTS/01_data_preparation/metadata/raw_dataset.json`
- `BraTS/02_preprocess/metadata/dataset.json`

产物是什么：

- nnU-Net 风格的训练图像目录
- nnU-Net 风格的标签目录
- 数据集任务定义 `dataset.json`

#### `plan-preprocess`

从哪里读：

- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/imagesTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/labelsTr`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/dataset.json`

写到哪里：

- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/dataset_fingerprint.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans_2d`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans_3d_fullres`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/splits_final.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations`

顺手同步到哪里：

- `BraTS/02_preprocess/metadata/dataset_fingerprint.json`
- `BraTS/02_preprocess/metadata/ProjectPlans.json`
- `BraTS/02_preprocess/metadata/splits_final.json`

详细日志写到哪里：

- `BraTS/02_preprocess/logs/preprocess_*.log`
- `BraTS/02_preprocess/logs/latest.log`

产物是什么：

- 数据集统计 fingerprint
- 自动规划出来的训练 plans
- 训练器真正读取的 `.npz + .pkl` 预处理样本
- 交叉验证分折文件
- 用于后续评估的 GT 目录

#### `train --fold 0`

从哪里读：

- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans.json`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/ProjectPlans_3d_fullres`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/splits_final.json`

写到哪里：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0`

里面会有什么：

- `checkpoint_best.pth`
- `checkpoint_final.pth`
- `checkpoint_latest.pth`
- `debug.json`
- `progress.png`
- `training_log_*.txt`
- `validation/`

顺手同步到哪里：

- `BraTS/03_training_and_results/results/fold_0`

产物是什么：

- 当前 fold 的权重
- 当前 fold 的训练日志和曲线
- 当前 fold 的 validation 预测结果

#### `train-all --npz`

从哪里读：

- 和单 fold 训练相同

写到哪里：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<trainer>__ProjectPlans__3d_fullres/fold_0 ... fold_4`

额外重要产物：

- 每个 `fold_x/validation` 里会多出 `.npz` 概率文件

为什么重要：

- `find-best-config`
- ensemble
- 后处理搜索

#### `find-best-config`

从哪里读：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/*/fold_x/validation`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations`

写到哪里：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/<model>/crossval_results_folds_*`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_information.json`
- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_instructions.txt`

产物是什么：

- 多 fold 汇总结果
- 最佳单模型或 ensemble 的选择结果
- 后处理建议
- 后续推理该用什么 trainer/plans/configuration/folds 的说明

#### `predict`

默认从哪里读模型：

- `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/...`
- 如果你没显式指定模型参数，优先看
  `nnUNet_test/nnUNet_results/Dataset220_BraTS2020/inference_information.json`

默认从哪里读输入：

- `BraTS/04_inference_and_evaluation/input`

默认写到哪里：

- `BraTS/04_inference_and_evaluation/predictions`

还会顺手写什么：

- `BraTS/04_inference_and_evaluation/predictions/dataset.json`
- `BraTS/04_inference_and_evaluation/predictions/plans.json`
- `BraTS/04_inference_and_evaluation/predictions/predict_from_raw_data_args.json`

注意：

- `predict` 默认只写推理产物和推理元数据
- `summary.json` 这类评估结果默认不再写到 `predictions/`

如果输入目录是空的，会发生什么：

- CLI 会从 `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/imagesTr` 随机抽样少量训练病例
- 把这批病例复制到 `BraTS/04_inference_and_evaluation/input`
- 抽样记录写到 `BraTS/04_inference_and_evaluation/input/sample_selection.json`

#### `evaluate`

从哪里读预测：

- 默认读 `BraTS/04_inference_and_evaluation/predictions`

从哪里读 GT：

- 默认读 `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations`

从哪里读元数据：

- 优先读 `BraTS/04_inference_and_evaluation/predictions/dataset.json`
- 优先读 `BraTS/04_inference_and_evaluation/predictions/plans.json`
- 缺失时回退到对应 trained model 目录里的 `dataset.json/plans.json`

写到哪里：

- 默认写 `BraTS/04_inference_and_evaluation/evaluation/summary.json`
- 如果你显式传 `--output-file`，就写到你指定的位置

产物是什么：

- `summary.json` 一类的结构化评估结果
- 当前预测子集或全量预测的 Dice 等指标

#### `python BraTS/run.py report-evaluation`

读取：

- `BraTS/04_inference_and_evaluation/evaluation/summary.json`
- `BraTS/04_inference_and_evaluation/predictions/*.nii.gz`
- `nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations/*.nii.gz`
- `nnUNet_test/nnUNet_raw/Dataset220_BraTS2020/imagesTr/*.nii.gz`
- 如果存在，还会读取 `BraTS/04_inference_and_evaluation/input/sample_selection.json`

写入：

- `BraTS/04_inference_and_evaluation/report/report.md`
- `BraTS/04_inference_and_evaluation/report/mean_dice_by_region.png`
- `BraTS/04_inference_and_evaluation/report/case_ranking.png`
- `BraTS/04_inference_and_evaluation/report/case_region_heatmap.png`
- `BraTS/04_inference_and_evaluation/report/best_case_overlay.png`
- `BraTS/04_inference_and_evaluation/report/worst_case_overlay.png`
- `BraTS/04_inference_and_evaluation/report/summary.copy.json`
- `BraTS/04_inference_and_evaluation/report/sample_selection.copy.json`

### 第一步：检查环境

```bash
python BraTS/run.py doctor
```

快速冒烟测试也可以直接跑：

```bash
python -m unittest BraTS.tests.test_smoke
```

当前快速冒烟测试覆盖：

- `python BraTS/run.py` 无参数时能正常显示帮助
- `doctor` 和 `doctor.py` 包装入口能正常执行
- `prepare-dataset` 可重复执行
- `prepare-dataset --help` 包含 `--force`
- `plan-preprocess --help` 包含复用/重算开关
- `plan-preprocess / train / predict / evaluate / find-best-config / report-evaluation --help` 能正常解析

`doctor` 本身检查内容：

- 工作区定位是否正确
- `project_config.json` 是否可解析
- raw/preprocessed/results 根目录是否存在
- `torch` 是否已安装
- CUDA 是否可见

### 第二步：先看懂第一个病例

```bash
python BraTS/run.py visualize-first-case
```

目的：

- 先确认数据本身没有把你绕晕
- 先建立 MRI 多模态与标签的空间直觉

### 第三步：准备训练目录

```bash
python BraTS/run.py prepare-dataset
```

结果：

- 第一次运行时构建 `PROJECT_RAW/Dataset220_BraTS2020`
- 如果目标目录已经完整存在，则直接复用并成功退出
- 如需强制重建，使用 `python BraTS/run.py prepare-dataset --force`

### 第四步：规划与预处理

```bash
python BraTS/run.py plan-preprocess
```

结果：

- 如果缺少 `dataset_fingerprint.json`，就生成它
- 如果缺少 `ProjectPlans.json`，就生成它
- 如果缺少某个 configuration 的预处理输出，就生成它
- 已存在的 fingerprint / plans / preprocess 输出默认直接复用

如需强制重算，可按需使用：

```bash
python BraTS/run.py plan-preprocess --recompute-fingerprint
python BraTS/run.py plan-preprocess --recompute-plans
python BraTS/run.py plan-preprocess --force-preprocess
python BraTS/run.py plan-preprocess --clean
```

预处理时每个 case 的详细输出现在默认写到：

- `BraTS/02_preprocess/logs/preprocess_*.log`
- `BraTS/02_preprocess/logs/latest.log`

终端默认只保留阶段信息和进度条。

### 第五步：先跑 `fold_0`

```bash
python BraTS/run.py train --fold 0
```

当前默认行为：

- 默认 trainer 仍然是 `SegTrainer + ProjectPlans + 3d_fullres`
- 默认训练轮数已经改成 `60 epochs`
- 如果该 fold 已经存在 checkpoint，`train` 会默认自动续训
- 只有显式加 `--restart-training` 才会忽略已有 checkpoint 从头开始

目的：

- 验证训练链路是否完整
- 排除路径、plans、GPU、日志写入、checkpoint 等工程问题

### 第六步：再跑完整五折

```bash
python BraTS/run.py train-all --npz
```

为什么建议带 `--npz`：

- 后面 `find-best-config` 和 ensemble 需要 validation 概率输出

### 第七步：选择最佳配置

```bash
python BraTS/run.py find-best-config
```

这一步会：

- 汇总多折结果
- 默认优先比较当前项目的默认训练组合
- 如果默认组合还没有可用的 `fold_x/validation` 结果，会自动回退到当前数据集下实际已有 validation 输出的模型
- 如果只完成了部分 fold，会自动只使用当前实际存在 validation 输出的 folds
- 需要时可通过 `--configurations/--trainers/--plans-identifiers` 比较多组组合
- 需要时比较 ensemble
- 自动寻找最合适的后处理规则

### 第八步：正式推理

```bash
python BraTS/run.py predict
```

当前默认行为：

- 如果你没有显式传 `--trainer/--configuration/--plans/--folds`，`predict` 会优先读取 `find-best-config` 生成的 `inference_information.json`
- 也就是说，它会默认使用当前已经选出来的最佳单模型，而不是死用项目初始默认 trainer
- 如果 `inference_information.json` 指向的是 ensemble，则需要你显式指定模型参数或按提示分别运行各成员模型
- 如果默认输入目录 `BraTS/04_inference_and_evaluation/input` 是空的，CLI 会自动从 `PROJECT_RAW/Dataset220_BraTS2020/imagesTr` 随机抽样 `8` 个训练病例，把它们当成临时验证集来跑推理
- 这批自动抽样病例会记录在 `BraTS/04_inference_and_evaluation/input/sample_selection.json`
- 如需显式控制抽样数量和随机种子，可使用 `--sample-training-cases` 和 `--sample-seed`
- 如需关闭这层默认自动抽样，可使用 `--disable-auto-sample-training`

输入默认来自：

- `BraTS/04_inference_and_evaluation/input`

输出默认写到：

- `BraTS/04_inference_and_evaluation/predictions`

### 第九步：正式评估

```bash
python BraTS/run.py evaluate
```

当前默认行为：

- 如果预测目录为空，会直接提示你先跑 `predict`
- 如果预测目录只覆盖了 ground truth 的一个子集，`evaluate` 会自动按这个子集评估，不再因为“不是全量 368 例”直接报错
- 在未显式指定模型元数据参数时，`evaluate` 也会优先使用 `find-best-config` 选出的最佳模型元数据
- 如果 `predict` 前一步用的是训练集随机抽样输入，`evaluate` 会继续按这批病例名做子集评估，不要求全量 368 例预测都存在

输出通常是：

- `BraTS/04_inference_and_evaluation/evaluation/summary.json`
- 按 case / 按 region 的结构化指标

### 第十步：自动生成评估报告和可视化

```bash
python BraTS/run.py report-evaluation
```

输出默认写到：

- `BraTS/04_inference_and_evaluation/report`

这里会生成：

- `report.md`
- 按 region 的 Dice 柱状图
- 当前评估子集的病例排序图
- 按病例 / 按 region 的 Dice heatmap
- WT / TC / ET 三张单独的病例排序图
- 最好病例和最差病例的 overlay 图
- Precision / Recall 图
- 预测体积相对 GT 的 volume bias 图
- `analysis.json`
- `case_metrics.csv`
- `case_analysis.csv`
- `case_analysis.md`
- `cases/` 逐病例 Markdown 和 overlay 图

这些 overlay 会直接把 MRI 底图、GT、预测和误差区域画出来，作用就是帮助你判断“哪里分得好、哪里分得差”。

---

## 5. 推荐阅读顺序

如果你是第一次真正通读这套项目，建议按下面顺序：

1. [00_first_case_visualization/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/00_first_case_visualization/README.md)
2. [01_data_preparation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/README.md)
3. [01_data_preparation/docs/data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/docs/data_contract.md)
4. [02_preprocess/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/README.md)
5. [02_preprocess/docs/preprocess_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/docs/preprocess_guide.md)
6. [03_training_and_results/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/README.md)
7. [03_training_and_results/docs/training_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/docs/training_guide.md)
8. [04_inference_and_evaluation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/README.md)
9. [04_inference_and_evaluation/docs/inference_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/docs/inference_guide.md)
10. [TERMS_AND_THEORY.md](/home/Creeken/Desktop/machine-learning-test/BraTS/TERMS_AND_THEORY.md)
11. [PIPELINE_EXPLANATION.md](/home/Creeken/Desktop/machine-learning-test/BraTS/PIPELINE_EXPLANATION.md)
12. [explain.md](/home/Creeken/Desktop/machine-learning-test/BraTS/explain.md)
13. [REFERENCES.md](/home/Creeken/Desktop/machine-learning-test/BraTS/REFERENCES.md)

推荐这样读的原因不是“从浅到深”这么简单，而是为了让你先建立：

- 数据契约
- plans 语义
- 训练目录结构
- 推理依赖关系

之后再去看 `trainer` 和 `predictor` 源码，脑子里才不会只剩函数调用。

---

## 6. 你最应该先掌握的四个代码文件

如果当前只能挑 4 个最关键的源码文件先吃透，建议优先看：

1. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
2. [default_experiment_planner.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/experiment_planners/default_experiment_planner.py)
3. [default_preprocessor.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/preprocessing/preprocessors/default_preprocessor.py)
4. [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)

这四个文件几乎对应整套系统的四根主梁：

- `cli.py`
  - 调度层
- `default_experiment_planner.py`
  - 自动规划层
- `default_preprocessor.py`
  - 数据变换层
- `nnUNetTrainer.py`
  - 训练主控层

---

## 7. 常见误解

### 误解 1：这就是一个普通训练脚本项目

不是。

这里的主角不是单个 `train.py`，而是一整条工程流水线。训练只是中间一环。

### 误解 2：只要知道训练命令就够了

不够。

这个项目里很多结果好坏，前因都在更早阶段：

- 数据转换是否正确
- labels 是否映射正确
- plans 是否解释合理
- preprocess 是否和 plans 一致

### 误解 3：`ProjectPlans.json` 只是个参数文件

不只是。

它是 planner 针对当前数据集做出的正式配置决议，训练、预处理、推理都在依赖它。

### 误解 4：`find-best-config` 只是锦上添花

不是。

它在这个项目里承担“训练结果到最终推理方案”的桥梁作用。

### 误解 5：先读 trainer 最省事

通常相反。

如果你没先理解数据契约和 plans，trainer 里的很多逻辑会显得毫无上下文。

---

## 8. 下一步应该去哪

- 想先理解数据：
  去看 [00_first_case_visualization/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/00_first_case_visualization/README.md)
- 想先理解训练输入如何构造：
  去看 [01_data_preparation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/README.md) 和 [data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/docs/data_contract.md)
- 想先理解 nnU-Net 自动规划：
  去看 [02_preprocess/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/README.md)
- 想先理解训练产物和 checkpoint：
  去看 [03_training_and_results/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/README.md)
- 想先理解正式推理和后处理：
  去看 [04_inference_and_evaluation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/README.md)
- 想按代码维度通读整个项目：
  去看 [explain.md](/home/Creeken/Desktop/machine-learning-test/BraTS/explain.md)

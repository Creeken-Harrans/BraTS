# BraTS 项目代码总解释

命令位置说明：
- 本文默认假设你当前目录就是 `BraTS` 项目根目录，因此命令示例写成 `python run.py ...`。
- 如果你当前在上一级目录 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径默认也都相对于 `BraTS` 项目根目录。

这份 `explain.md` 的目标不是复述 README，而是把 `/home/Creeken/Desktop/machine-learning-test/BraTS/` 里真正会参与运行、组织流程、产出结果的代码和关键文件逐层讲清楚。

你可以把它理解成：

- 顶层 README 之上的“源码总图”
- 各阶段 README / guide 之上的“代码层总解释”

如果你只想快速找文档入口：

- 看项目整体：`README.md`
- 看概念：`TERMS_AND_THEORY.md`
- 看真实流程：`PIPELINE_EXPLANATION.md`
- 看某个阶段怎么跑：各阶段 `README.md` 和 `docs/*.md`
- 看源码分工：这份 `explain.md`

我先给一个结论：

- 这个项目本质上是一个“为当前 BraTS2020 数据和当前目录结构本地化过的 nnU-Net 派生项目”。
- 它没有额外发明新算法，核心思想仍然是 nnU-Net：`dataset -> fingerprint -> plans -> preprocess -> trainer -> validation -> inference -> ensemble/postprocessing/evaluation`。
- 项目里有两类代码：
  - 项目外壳代码：为了适配你的工作区路径、命令入口、BraTS 数据整理和文档导出而写的本地代码。
  - 核心训练代码：大部分来自 nnU-Net 风格实现，负责实验规划、预处理、训练、推理、评估、后处理。

如果你只想先抓主线，可以先记住下面这个调用链：

```text
run.py
  -> brats_project.cli.main()
     -> doctor / visualize-first-case / prepare-dataset / plan-preprocess / train / predict / evaluate ...
        -> 对应模块 API
           -> experiment_planning / preprocessing / training / inference / evaluation / postprocessing
```

---

## 1. 顶层目录到底在分什么层

### `00_first_case_visualization/`

作用：先把 BraTS 原始病例“看懂”。

- 这里不是训练必要步骤，但它解决一个非常现实的问题：医学影像不是 JPG/PNG，而是带几何信息的 3D NIfTI 体数据。
- 所以这一层的代码主要在做：
  - 递归寻找第一个合法病例目录
  - 识别 `t1/t1ce/t2/flair/seg`
  - 校验它们是否同空间对齐
  - 生成文字摘要、统计 CSV、以及多张用于理解数据的 PNG

### `01_data_preparation/`

作用：把原始 BraTS 目录转换成 nnU-Net/本项目能训练的数据集目录。

- 输入是官方/原始 BraTS 病例目录。
- 输出是 `PROJECT_RAW/Dataset220_BraTS2020/`。
- 这一层最重要的是：
  - 通道命名改成 `_0000/_0001/_0002/_0003`
  - 标签从 BraTS 原始编码映射到项目训练编码
  - 生成 `dataset.json`
  - 如果目标目录已经完整存在，则直接复用；只有 `--force` 才重建

### `02_preprocess/`

作用：生成指纹、plans、split，并把原始图像转成训练期快速读取的预处理格式。

- fingerprint：从数据统计出 spacing、裁剪后形状、前景强度统计等信息。
- plans：根据 fingerprint 自动推导 patch size、batch size、网络配置、归一化和重采样策略。
- preprocess：真正把病例变成训练直接读取的预处理数组/压缩块文件。

### `03_training_and_results/`

作用：项目真正的核心代码所在地。

- `src/brats_project/` 是主包。
- 这里几乎包含整个训练、推理、评估体系。
- `results/` 放训练期样例结果和日志快照。

### `04_inference_and_evaluation/`

作用：推理阶段的输入/输出目录和说明文档。

- 这一层更多是工程落点和文档，不是核心算法代码。

### `Document/`

作用：把项目 Markdown 文档批量导出成 PDF。

- 和模型训练无关。
- 是文档生产工具层。

---

## 2. 顶层入口文件

### `run.py`

这是整个项目最薄的一层入口。

它只做三件事：

1. 找到项目根目录。
2. 把 `03_training_and_results/src` 放进 `sys.path`。
3. 调用 `brats_project.cli.main()`。

也就是说：

- `run.py` 本身没有业务逻辑。
- 它只是把这个本地项目包装成一个可直接 `python run.py ...` 调用的程序。

### `pyproject.toml`

这个文件定义了项目如何被安装成 Python 包，以及有哪些控制台命令。

关键内容：

- 依赖列表：
  - `torch`
  - `SimpleITK`
  - `nibabel`
  - `batchgenerators`
  - `dynamic-network-architectures`
  - `blosc2`
  - `matplotlib/pandas/scikit-image/scikit-learn` 等
- 暴露的命令：
  - `brats`
  - `brats_plan_and_preprocess`
  - `brats_train`
  - `brats_predict`
  - `brats_find_best_configuration`
  - `brats_apply_postprocessing`
  - `brats_ensemble`
  - `brats_evaluate_folder`

所以 `run.py` 和安装后的 `brats` 命令，本质上是同一套 CLI 的两种入口方式。

### `project_config.json`

这是本项目的本地约定总配置。

它不控制模型结构细节，主要控制三类默认值：

- 数据集默认值：
  - `id=220`
  - `name=Dataset220_BraTS2020`
  - `trainer=SegTrainer`
  - `plans=ProjectPlans`
  - `default_configuration=3d_fullres`
- 路径默认值：
  - 原始 BraTS 根目录
  - raw/preprocessed/results 根目录
  - 可视化输出目录
  - 推理输入输出目录
- 运行环境默认值：
  - 预期 `torch/cuda`
  - 默认进程数

这个文件的核心价值是把本地路径和默认参数收口到单点，不让路径常量散落在各模块。

---

## 3. CLI 层是怎么把命令转成代码调用的

### `03_training_and_results/src/brats_project/cli.py`

这是项目真正的控制台总入口。

它做的事情可以概括成五类：

1. 读取项目配置和路径
2. 配置环境变量
3. 解析子命令
4. 调用对应 API
5. 同步关键元数据到项目文档目录

它的重要辅助函数：

- `_load_external_module`：动态加载外部脚本，比如 `visualize_first_case.py` 和 `prepare_brats2020_for_project.py`
- `_sync_preprocess_metadata`：把 `dataset.json / dataset_fingerprint.json / splits_final.json / plans.json` 同步到 `02_preprocess/metadata/`
- `_sync_training_snapshot`：把训练输出里的 `debug.json/progress.png/log` 拷到 `03_training_and_results/results/fold_x/`
- `_build_device`：把 `cpu/cuda/mps` 字符串转成 `torch.device`

它暴露的核心命令：

- `doctor`
  - 检查项目根、工作区根、路径配置、torch 和 CUDA 可见性
- `visualize-first-case`
  - 调用 `00_first_case_visualization/visualize_first_case.py`
- `prepare-dataset`
  - 调用 `01_data_preparation/scripts/prepare_brats2020_for_project.py`
- `plan-preprocess`
  - 默认按需调用 `extract_fingerprints -> plan_experiments -> preprocess`
  - 已存在的 fingerprint / plans / preprocess 输出会直接复用
  - 只有显式传入 `--recompute-fingerprint / --recompute-plans / --force-preprocess / --clean` 才重做
- `train`
  - 调用 `run_training.run_training`
- `train-all`
  - 遍历默认 5 folds
- `find-best-config`
  - 调用 `evaluation.find_best_configuration`
- `predict`
  - 转发到推理入口
- `evaluate`
  - 调用评估入口

你可以把 `cli.py` 理解为“项目调度中心”。

### `project_layout.py`

这个文件负责“项目如何找到自己”和“路径如何解释”。

主要职责：

- `get_project_root()`：通过向上找 `project_config.json` 定位项目根
- `get_workspace_root()`：优先向上找名为 `machine-learning-test` 的工作区目录
- `load_project_config()`：读取 `project_config.json`
- `resolve_workspace_path()`：把相对路径解释成相对于 workspace 的绝对路径
- `configure_environment()`：
  - 设定 `PROJECT_RAW`
  - 设定 `PROJECT_PREPROCESSED`
  - 设定 `PROJECT_RESULTS`
  - 设定默认并发环境变量

这个文件的工程价值非常高，因为它让所有模块不必自己硬编码路径解析逻辑。

### `paths.py`

这是更底层的路径常量层。

作用：

- 从环境变量或 `project_layout` 默认值里导出：
  - `PROJECT_RAW`
  - `PROJECT_PREPROCESSED`
  - `PROJECT_RESULTS`

所以：

- `project_layout.py` 负责“设置环境”
- `paths.py` 负责“给其他模块统一读取环境”

### `configuration.py`

这是很薄的全局配置层。

只定义：

- `default_num_processes`
- `ANISO_THRESHOLD`
- `default_n_proc_DA`

它更多是给各模块导出少量全局常量。

---

## 4. 第一阶段：理解原始病例的脚本

### `00_first_case_visualization/visualize_first_case.py`

这是一个很完整的“病例理解脚本”。

它的职责可以拆成 8 段：

1. 找工作区和输出目录
2. 在 BraTS 根目录里递归找第一个合法病例目录
3. 自动识别每个 NIfTI 文件属于 `t1/t1ce/t2/flair/seg`
4. 用 `nibabel` 载入 3D 体数据
5. 校验所有模态和分割是否 shape/spacing/affine 对齐
6. 统计标签和强度信息
7. 生成多种可视化图
8. 写说明 README

这个文件的重要函数分组如下。

#### 4.1 路径与文件发现

- `find_workspace_root`
- `resolve_workspace_path`
- `find_case_candidate_dirs`
- `select_first_case_dir`
- `discover_case_files`
- `detect_series_role`

作用：

- 从磁盘里自动发现“第一个合法的 BraTS 病例目录”
- 识别哪个文件是 `t1`、哪个是 `seg`
- 如果目录不完整或角色冲突，直接报错

#### 4.2 体数据载入与对齐验证

- `load_volume`
- `validate_volume_alignment`

作用：

- 对 MRI 模态按 float 读入，对 `seg` 按整数标签读入
- 确保所有 volume 在同一空间，否则 overlay 会失真

#### 4.3 文本/JSON/CSV 摘要生成

- `compute_seg_label_stats`
- `build_case_summary`
- `write_case_summary_text`
- `compute_intensity_stats`
- `write_seg_labels_summary`

作用：

- 写 `case_summary.txt/json`
- 写 `intensity_stats.csv`
- 写 `seg_labels_summary.txt`

#### 4.4 切片与显示辅助

- `compute_display_range`
- `get_plane_length`
- `get_display_slice`
- `count_nonzero_per_slice`
- `select_montage_indices`
- `select_best_seg_slice`

作用：

- 统一 axial/coronal/sagittal 三视图切片逻辑
- 自动选择更有信息量的切片，而不是全都盯着中间层

#### 4.5 标签着色和图例

- `build_seg_colormap`
- `build_label_handles`

作用：

- 给离散标签定义颜色映射和 legend

#### 4.6 可视化产物生成

- `plot_modalities_mid_slices`
- `plot_modality_montage`
- `plot_segmentation_montage`
- `plot_overlay_best_slices`
- `plot_tumor_bbox_views`
- `plot_intensity_histograms`
- `plot_seg_label_distribution`

作用：

- 生成所有 PNG 输出

#### 4.7 肿瘤框选逻辑

- `compute_3d_bounding_box`
- `project_bbox_to_plane`
- `crop_display_slice`

作用：

- 从 3D seg 提取肿瘤包围盒，并投影到三视图上

#### 4.8 输出 README

- `write_readme`

这个函数不是训练代码，但很实用：

- 它自动给输出目录写一份面向学习者的 BraTS 数据解说文档。

整体评价：

- 这是项目里最“教学友好”的脚本。
- 它几乎不参与后续训练，但非常适合作为你理解 3D 医学影像输入的第一站。

---

## 5. 第二阶段：把原始 BraTS 数据转换成训练目录

### `01_data_preparation/scripts/prepare_brats2020_for_project.py`

这个脚本是 raw dataset 构建器。

它的核心工作流：

1. 找到原始 BraTS 根目录
2. 扫描每个病例目录
3. 找出 `t1/t1ce/t2/flair/seg`
4. 校验几何对齐
5. 复制四个模态到 `imagesTr`
6. 把标签从 BraTS 编码改为训练编码
7. 生成 `dataset.json`

关键函数：

- `find_case_file`
  - 支持 `.nii.gz` 和 `.nii`
- `validate_case_geometry`
  - 校验 size/spacing/origin/direction 全一致
- `collect_cases`
  - 收集所有合法病例
- `write_image_as_niigz`
  - 重写模态文件到目标目录
- `convert_brats_seg_to_project`
  - 标签映射：
    - `0 -> 0`
    - `2 -> 1`
    - `1 -> 2`
    - `4 -> 3`

为什么这样映射？

- 原始 BraTS 标签是 `0/1/2/4`
- 训练和 region-based handling 更喜欢连续标签体系
- 所以这个脚本先把原标签映射到训练内部标签，再在 `dataset.json` 里声明 region 语义：
  - `whole_tumor = [1,2,3]`
  - `tumor_core = [2,3]`
  - `enhancing_tumor = [3]`

也就是说：

- 真正训练时并不直接把标签理解为“4 类 softmax 分类”
- 而是利用 region-based target 定义 BraTS 常见的三个区域目标

这个脚本输出的数据契约由下面这些文件共同固定：

- `01_data_preparation/metadata/raw_dataset.json`
- `01_data_preparation/docs/data_contract.md`

---

## 6. 第三阶段：实验规划与预处理

这部分代码在 `brats_project.experiment_planning` 和 `brats_project.preprocessing`。

### `experiment_planning/plan_and_preprocess_api.py`

这是上层 API 封装。

它把预处理阶段拆成三个显式步骤：

- `extract_fingerprints`
- `plan_experiments`
- `preprocess`

每一步都可以单独调用，也可以在 CLI 里串起来。

### `experiment_planning/plan_and_preprocess_entrypoints.py`

这是这些 API 的命令行包装层。

它暴露：

- `extract_fingerprint_entry`
- `plan_experiment_entry`
- `preprocess_entry`
- `plan_and_preprocess_entry`

这个文件本身算法很少，主要作用是参数解析和命令桥接。

### `experiment_planning/dataset_fingerprint/fingerprint_extractor.py`

这个文件负责“看数据长什么样”，也就是 fingerprint 提取。

它对每个病例做的事：

1. 读图像和分割
2. 执行非零区域裁剪
3. 统计裁剪后 shape
4. 记录 spacing
5. 从前景区域采样强度
6. 汇总整个数据集的强度统计

输出到：

- `dataset_fingerprint.json`

这个 fingerprint 主要被后面的 planner 用来决定：

- target spacing
- patch size
- batch size
- 归一化策略

### `experiment_planning/verify_dataset_integrity.py`

作用：在真正做 fingerprint/preprocess 前验证数据合法性。

它会检查：

- 图像和分割是否匹配
- 通道数是否符合 `dataset.json`
- 标签值是否都在声明范围里
- 各模态几何是否一致

这是“早崩溃”模块，防止脏数据拖到训练才报错。

### `experiment_planning/experiment_planners/default_experiment_planner.py`

这是整个自动规划逻辑最核心的文件之一。

它负责把 fingerprint 变成 plans。

它主要决定：

- fullres target spacing
- transpose 方向
- 2D/3D 配置是否存在
- 低分辨率配置是否需要
- patch size
- batch size
- 网络层级结构
- 重采样函数
- 归一化方案

重要思想：

- 先根据 spacing 推断合理 patch 的各向异性比例
- 再根据显存目标不断缩 patch，直到估计的 VRAM 占用满足预算
- 再据此给出 batch size

输出就是 `ProjectPlans.json` 或指定名字的 plans 文件。

### `experiment_planning/experiment_planners/network_topology.py`

作用：根据 patch size 和 spacing 计算 pooling/conv 的层级结构。

它给 planner 提供：

- 每个 stage 的 stride
- kernel size
- 网络深度约束

### `experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py`

作用：定义 residual encoder UNet 风格的 planner 变体。

文件里几个类：

- `ResEncUNetPlanner`
- `nnUNetPlannerResEncM`
- `nnUNetPlannerResEncL`
- `nnUNetPlannerResEncXL`

这些类是更偏“架构族配置模板”，不是你的默认主线，但保留了替换 planner 的能力。

### `experiment_planning/experiment_planners/resencUNet_planner.py`

作用：给 residual encoder UNet 规划器做薄封装。

### `experiment_planning/experiment_planners/resampling/planners_no_resampling.py`

作用：定义“不做重采样”的 planner 变体。

### `experiment_planning/experiment_planners/resampling/resample_with_torch.py`

作用：定义使用 torch 重采样的 planner 变体。

### `experiment_planning/plans_for_pretraining/move_plans_between_datasets.py`

作用：把某个数据集上的 plans 搬运到另一个数据集，偏向预训练/迁移实验场景。

### `preprocessing/preprocessors/default_preprocessor.py`

这是预处理阶段最核心的另一个文件。

单个病例会走的顺序是：

1. 把数据转成 float32
2. 根据 plans 执行轴转置
3. 基于非零区域裁剪
4. 计算目标 shape
5. 先做归一化
6. 再做重采样
7. 如果有 seg，则抽前景位置做训练采样辅助
8. 存成预处理后的 dataset case

关键点：

- 归一化在重采样前做
- `class_locations` 会写入 `properties`，供 dataloader 过采样前景 patch
- 输出采用 `blosc2` 保存，以提高训练期随机 patch 读取效率

重要函数：

- `run_case_npy`
- `run_case`
- `run_case_save`
- `_sample_foreground_locations`
- `_normalize`
- `run`

### `preprocessing/cropping/cropping.py`

作用：生成非零 mask 并裁剪到非零区域。

函数很少，但很关键：

- `create_nonzero_mask`
- `crop_to_nonzero`

### `preprocessing/resampling/default_resampling.py`

作用：封装默认重采样逻辑。

它解决的问题：

- 新 spacing 下 shape 应该变多少
- 各向异性强时是否单独处理 z 轴
- 图像和分割该分别用什么插值策略

### `preprocessing/resampling/resample_torch.py`

作用：用 torch 做重采样的实现。

### `preprocessing/resampling/no_resampling.py`

作用：不重采样时的占位实现。

### `preprocessing/resampling/utils.py`

作用：按名字查找重采样函数。

### `preprocessing/normalization/default_normalization_schemes.py`

作用：定义归一化方法族。

主要类：

- `ImageNormalization`
- `ZScoreNormalization`
- `CTNormalization`
- `NoNormalization`
- `RescaleTo01Normalization`
- `RGBTo01Normalization`

BraTS MRI 主线通常主要用到 z-score 风格归一化。

### `preprocessing/normalization/map_channel_name_to_normalization.py`

作用：把 channel 名称映射到某种 normalization class。

### `preprocessing/normalization/readme.md`

作用：解释归一化层的思路，不是代码，但有助于理解 planner 为什么这样选。

---

## 7. 数据 I/O 层

这部分负责“怎么把磁盘文件读成网络吃的数组”，以及“怎么把分割结果写回原格式”。

### `imageio/base_reader_writer.py`

这是抽象接口。

它强制每种 I/O 实现提供：

- `read_images`
- `read_seg`
- `write_seg`

统一约定：

- 图像必须被读成 `(c, x, y, z)` 形式
- metadata 里必须带 `spacing`

### `imageio/reader_writer_registry.py`

作用：根据 `dataset.json` 或文件后缀自动选择正确 I/O 类。

它维护的注册类包括：

- `NaturalImage2DIO`
- `SimpleITKIO`
- `Tiff3DIO`
- `NibabelIO`
- `NibabelIOWithReorient`

### `imageio/simpleitk_reader_writer.py`

这是医学影像主线最重要的 I/O 实现之一。

特点：

- 支持 `.nii.gz/.nrrd/.mha/.gipl`
- 用 SimpleITK 读取和写回
- 保留原始 spacing/origin/direction
- 在内部把 spacing 顺序变成项目统一的数组轴顺序

`SimpleITKIOWithReorient` 额外支持先重定向到指定 orientation 再处理。

### `imageio/nibabel_reader_writer.py`

作用：使用 nibabel 处理 NIfTI。

特点：

- 把 nibabel 的轴顺序转成和 SimpleITK 主线一致
- 写回时再转回原 affine

`NibabelIOWithReorient` 是带 reorient 的变体。

### `imageio/natural_image_reader_writer.py`

作用：处理普通自然图像数据。

对 BraTS 主线基本不重要，但它保留了框架对 2D 图像数据集的兼容性。

### `imageio/tif_reader_writer.py`

作用：处理 3D TIFF 数据。

对 BraTS 主线也不是核心。

---

## 8. Plans 和标签管理层

### `utilities/plans_handling/plans_handler.py`

这个文件是“plans 访问门面”。

它把 JSON 形式的 plans 包装成两个对象：

- `PlansManager`
- `ConfigurationManager`

它解决的问题：

- 配置继承展开
- 根据名字找到 preprocessor/resampling/reader_writer/label_manager 类
- 提供更好用的属性接口，而不是到处手搓 dict key

这层是整个项目非常关键的胶水层。

### `utilities/label_handling/label_handling.py`

这个文件负责统一处理“标签”和“region”两种训练语义。

核心类是 `LabelManager`。

它负责：

- 识别当前是否为 region-based training
- 维护 `foreground_labels` 和 `foreground_regions`
- 定义推理期非线性：
  - labels 用 softmax/argmax
  - regions 用 sigmoid + 阈值
- 把预测 logits/probabilities 转回 segmentation
- 把 label map 转成 one-hot
- 处理 ignore label

对于 BraTS，这层尤其重要，因为 BraTS 常用 whole tumor / tumor core / enhancing tumor 这种 region 目标。

### `utilities/get_network_from_plans.py`

作用：根据 plans 里声明的网络类名和参数实例化网络。

### `utilities/find_class_by_name.py`

作用：按字符串类名递归查找 Python 类。

很多“通过 plans 或 CLI 字符串指定类”的地方都依赖它。

### `utilities/json_export.py`

作用：把 numpy/torch 等不易 JSON 序列化的对象转成普通类型。

### `utilities/dataset_name_id_conversion.py`

作用：在 `220` 和 `Dataset220_BraTS2020` 之间双向转换。

### `utilities/file_path_utilities.py`

作用：统一模型输出目录命名和解析。

例如：

- `SegTrainer__ProjectPlans__3d_fullres`
- `fold_0`
- ensemble 输出命名

### `utilities/utils.py`

作用：各种数据集文件辅助函数。

最重要的是：

- 从 `imagesTr/labelsTr` 结构自动构造每个 case 的图像文件列表和标签文件路径

### `utilities/crossval_split.py`

作用：生成固定随机种子的五折 split。

### `utilities/collate_outputs.py`

作用：把训练/验证 step 的输出聚合起来。

### `utilities/default_n_proc_DA.py`

作用：决定数据增强进程数。

### `utilities/helpers.py`

作用：提供一些小工具，比如清缓存、dummy context、softmax helper 等。

### `utilities/network_initialization.py`

作用：网络初始化辅助类。

### `utilities/ddp_allgather.py`

作用：分布式训练时的 all-gather 梯度/对象同步辅助。

### `utilities/overlay_plots.py`

作用：生成 overlay 图。

这属于辅助可视化工具层，不是主训练链必要部分。

---

## 9. 数据加载与增强

### `training/dataloading/nnunet_dataset.py`

这个文件定义了训练时如何读取预处理后的 case。

核心抽象：

- `nnUNetBaseDataset`
- `nnUNetDatasetNumpy`
- `nnUNetDatasetBlosc2`

BraTS 这套项目默认主要走 `nnUNetDatasetBlosc2`。

为什么？

- 预处理结果当前默认以 `.npz + .pkl` 形式保存，以换取更稳的跨环境训练兼容性；训练时会按需解包为 `.npy`
- 读取时既比原始 NIfTI 快，也更适合随机 patch 采样

### `training/dataloading/data_loader.py`

这个文件是真正的 patch 采样 dataloader。

它做的事：

1. 随机选病例
2. 根据 `class_locations` 决定是否过采样前景
3. 从 3D case 中裁 patch
4. 必要时拼接 cascade 上一阶段分割
5. 应用训练增强 transform
6. 组 batch 返回

它为什么重要：

- 训练稳定性很大程度依赖这个文件的 patch 采样策略
- 没有它，前景稀少时模型很容易一直看到空背景 patch

### `training/dataloading/utils.py`

作用：把 `.npz` 解包成 `.npy`，供某些数据集格式使用。

### `training/data_augmentation/compute_initial_patch_size.py`

作用：根据旋转/缩放增强范围，估计增强前需要预留多大的 patch。

### `training/data_augmentation/custom_transforms/*.py`

这些文件定义自定义增强变换。

- `cascade_transforms.py`
  - 处理 cascade 任务里把上一阶段 seg 当额外输入，并对 one-hot seg 做随机形态扰动
- `deep_supervision_donwsampling.py`
  - 为 deep supervision 生成多尺度分割 target
- `masking.py`
  - 使用 mask 对指定通道做外部置零
- `region_based_training.py`
  - 把 segmentation 转换成 region target
- `transforms_for_dummy_2d.py`
  - 3D 数据做 dummy 2D augmentation 时在 2D/3D 之间转换

---

## 10. 训练主链

### `run/run_training.py`

这是训练入口函数层。

它负责：

- 根据 trainer 名称找到 trainer 类
- 读取 plans 和 dataset.json
- 构造 trainer
- 决定是新训练、继续训练、只验证还是加载预训练权重
- 单卡运行或 DDP 多卡运行

重要函数：

- `get_trainer_from_args`
- `maybe_load_checkpoint`
- `run_training`
- `run_ddp`

这里还有一个现在已经明确收口的约束：

- trainer 恢复只支持从磁盘上的 checkpoint 文件加载
- 不再保留“`load_checkpoint` 可以直接接收内存 dict”这种未真正实现的僵尸接口

当前项目里的实际默认行为还有两点需要特别记住：

- 默认训练轮数已经改成 `100`
- 如果当前 fold 已经存在 checkpoint，CLI 会优先把这次训练解释成“继续训练”
- 自动续训时会优先拿 `checkpoint_final`，其次才是 `checkpoint_latest` 和 `checkpoint_best`
- `checkpoint_latest` 现在按 epoch 持续覆盖更新，不再长时间停留在很旧的恢复点

### `run/load_pretrained_weights.py`

作用：把预训练权重安全加载进网络。

### `training/nnUNetTrainer/SegTrainer.py`

这是本项目默认 trainer 名字。

它没有额外逻辑，只是：

- `class SegTrainer(nnUNetTrainer)`

换句话说：

- `SegTrainer` 是项目默认 trainer 别名
- 真正逻辑全在 `nnUNetTrainer.py`

### `training/nnUNetTrainer/nnUNetTrainer.py`

这是整个训练系统最核心的文件。

它几乎包含全部训练生命周期管理逻辑：

- 初始化设备、plans、configuration、label manager
- 构建网络
- 构建 optimizer 和 lr scheduler
- 创建 dataloader
- 设置增强
- 训练 step
- 验证 step
- 日志记录
- checkpoint 保存/恢复
- 最终滑窗验证导出

你可以把它理解成“训练总控类”。

最重要的方法：

- `initialize`
  - 网络、优化器、loss、dataset class 初始化
- `_set_batch_size_and_oversample`
  - DDP 时按 worker 调整 batch 和过采样比例
- `_build_loss`
  - labels 和 regions 两种情况下分别构造 loss
- `configure_rotation_dummyDA_mirroring_and_inital_patch_size`
  - 决定旋转、镜像、dummy 2D DA 策略
- `get_dataloaders`
  - 组装训练/验证 dataloader 和 augmenter
- `get_training_transforms`
  - 定义训练增强流水线
- `get_validation_transforms`
  - 定义验证 transform
- `train_step`
  - 一个 batch 的前向、loss、反向、梯度裁剪、更新
- `validation_step`
  - 一个 batch 的在线验证和 pseudo dice 统计
- `perform_actual_validation`
  - 对真实验证集做滑窗推理并导出 segmentation，再计算 summary.json
- `run_training`
  - 完整 epoch 循环

### `training/logging/nnunet_logger.py`

作用：维护训练日志、EMA 指标和 `progress.png` 绘图。

### `training/loss/*.py`

这些文件定义损失函数。

- `dice.py`
  - Dice 相关实现
- `robust_ce_loss.py`
  - CrossEntropy 和 TopK loss 变体
- `compound_losses.py`
  - Dice + CE / Dice + BCE / Dice + TopK 等组合损失
- `deep_supervision.py`
  - Deep supervision wrapper

### `training/lr_scheduler/*.py`

- `polylr.py`
  - 默认 poly LR scheduler
- `warmup.py`
  - warmup 调度器实现

### `training/nnUNetTrainer/variants/*`

这一整组文件是 trainer 变体库。

它们的意义不是“项目主线必须用”，而是保留对 nnU-Net 风格实验变体的兼容。

#### 数据增强变体

- `variants/data_augmentation/nnUNetTrainerDA5.py`
  - 更复杂的数据增强方案
- `variants/data_augmentation/nnUNetTrainerDAOrd0.py`
  - 使用不同插值阶数的数据增强变体
- `variants/data_augmentation/nnUNetTrainerNoDA.py`
  - 关闭数据增强
- `variants/data_augmentation/nnUNetTrainer_noDummy2DDA.py`
  - 不使用 dummy 2D augmentation
- `variants/data_augmentation/nnUNetTrainerNoMirroring.py`
  - 禁用或限制 mirroring

#### 损失变体

- `variants/loss/nnUNetTrainerCELoss.py`
  - CE loss trainer
- `variants/loss/nnUNetTrainerDiceLoss.py`
  - Dice loss trainer
- `variants/loss/nnUNetTrainerTopkLoss.py`
  - TopK loss trainer

#### 学习率调度变体

- `variants/lr_schedule/nnUNetTrainer_warmup.py`
  - warmup 版 trainer
- `variants/lr_schedule/nnUNetTrainerCosAnneal.py`
  - cosine annealing 版 trainer

#### 网络结构变体

- `variants/network_architecture/nnUNetTrainerBN.py`
  - batch norm 版本
- `variants/network_architecture/nnUNetTrainerNoDeepSupervision.py`
  - 关闭 deep supervision

#### 优化器变体

- `variants/optimizer/nnUNetTrainerAdam.py`
  - Adam 版
- `variants/optimizer/nnUNetTrainerAdan.py`
  - Adan 版

#### 采样变体

- `variants/sampling/nnUNetTrainer_probabilisticOversampling.py`
  - 概率式前景过采样

#### 训练长度变体

- `variants/training_length/nnUNetTrainer_Xepochs.py`
  - 1/5/10/20/.../8000 epochs 变体
- `variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py`
  - 指定 epoch 且无 mirroring

#### benchmark/competition/primus

- `variants/benchmarking/*.py`
  - 跑 5 epoch 等快速 benchmark
- `variants/competitions/aortaseg24.py`
  - 某比赛特化 trainer 组合
- `primus/primus_trainers.py`
  - Primus 系列 trainer 变体

对你这份 BraTS 项目当前默认配置来说，这些文件大部分是“保留能力”，不是主线入口。

---

## 11. 推理主链

### `inference/predict_from_raw_data.py`

这是推理系统最核心的文件。

核心类：`nnUNetPredictor`

它负责：

1. 从训练输出目录恢复模型
2. 载入 plans、dataset.json、fold 权重
3. 把原始图像送入 preprocessor
4. 执行滑窗推理
5. 多 fold 权重平均
6. 把 logits 导出成原始空间 segmentation

在当前项目的 CLI 封装里，`python run.py predict` 还有一层额外默认逻辑：

- 如果你没有显式传 `--trainer/--configuration/--plans/--folds`
- CLI 会优先读取 `PROJECT_RESULTS/Dataset220_BraTS2020/inference_information.json`
- 然后自动切换到 `find-best-config` 当前选出来的最佳单模型
- 如果默认输入目录 `../BraTS-Dataset/inference/input` 为空
- CLI 会进一步从 `PROJECT_RAW/Dataset220_BraTS2020/imagesTr` 随机抽样少量训练病例，把它们复制成临时验证输入
- 默认抽样数量是 `8`，随机种子默认是 `42`
- 抽样清单会落到 `04_inference_and_evaluation/metadata/sample_selection.json`

这意味着项目默认推理不再是“死用初始默认 trainer”，而是“优先跟随当前最佳模型结论”。

如果 `inference_information.json` 指向的是 ensemble，CLI 不会偷偷只跑其中一个成员，而是要求你显式指定模型参数。

关键方法：

- `initialize_from_trained_model_folder`
  - 从 `plans.json + dataset.json + checkpoint` 恢复 predictor
- `predict_from_files`
  - 批量从输入文件夹推理
- `predict_from_data_iterator`
  - 从预处理迭代器读取并执行预测
- `predict_logits_from_preprocessed_data`
  - 对单个预处理样本做模型推理，支持多 fold 参数平均
- `predict_sliding_window_return_logits`
  - 真正的滑窗推理入口
- `_internal_get_sliding_window_slicers`
  - 计算每个 tile 的切片窗口
- `_internal_maybe_mirror_and_predict`
  - test-time augmentation 镜像预测
- `_internal_predict_sliding_window_return_logits`
  - 聚合高斯权重 tile 预测结果

补充一个实现细节：

- 当前项目不再直接依赖私有的 `torch._dynamo.OptimizedModule`
- 对“一个 module 是否已经被 `torch.compile` 包装”以及“如何拿到原始 module”的判断，已经统一收敛到公共 helper
- 这样做的目的不是改功能，而是降低不同 PyTorch 版本下因为内部 API 变动导致导入失败的风险

另外，DDP 下的 compile 判定也已经修正：

- 现在不会再错误要求 `self.network.module` 先是 compiled module 才允许 compile
- 也就是说，只要环境变量允许且当前网络还没被 compile，DDP 路径也能正常进入 `torch.compile`

### `inference/sliding_window_prediction.py`

作用：

- 计算滑窗步长
- 计算 Gaussian 权重图

### `inference/data_iterators.py`

作用：

- 在推理时把“原始文件 -> 预处理结果”的过程做成迭代器
- 支持从文件和从内存数组两种来源预处理

### `inference/export_prediction.py`

作用：

- 把预测 logits/probabilities 转回原始空间
- 逆转预处理过程：
  - 逆重采样
  - 逆裁剪
  - 逆 transpose
- 最终写出 `.nii.gz` segmentation

---

## 12. 评估、集成与后处理

### `evaluation/evaluate_predictions.py`

这是评估模块主文件。

核心逻辑：

- 读 GT 和预测
- 对每个 label/region 计算：
  - TP / FP / FN / TN
  - Dice
  - IoU
- 汇总成 `summary.json`

在当前项目的 CLI 封装里，建议按下面这个可执行写法运行评估：

```bash
python run.py evaluate
```

它现在默认是严格模式：

- 如果预测目录为空，会直接报“先跑 predict”的清晰错误
- 如果预测文件名和 ground-truth 集不完全一致，也会直接报错，而不是静默跳过缺失病例
- 如果前一步 `predict` 用的是从训练集随机抽样出来的临时验证集，想评估这批样本也必须显式传 `--chill`

它支持：

- 基于 dataset/plans 自动推断 labels 与 IO
- 简单指定 labels 的轻量评估

### `evaluation/accumulate_cv_results.py`

作用：

- 把多个 fold 的 `validation` 预测复制到一个公共目录
- 重新评估，得到 crossval 汇总结果

### `evaluation/find_best_configuration.py`

这个文件的作用很重要：自动决定“哪套配置最好”。

它会：

1. 检查允许的 trained models 是否存在
2. 对每个单模型收集 crossval 结果
3. 可选地枚举两两 ensemble
4. 重新评估 ensemble
5. 找到最优者
6. 为最优者自动决定 postprocessing
7. 生成 `inference_information.json` 和推理说明文本

也就是说：

- 它是“从多个训练配置选最终推理方案”的自动决策器。

### `ensembling/ensemble.py`

作用：

- 读取多个 `.npz` 概率预测
- 对概率做平均
- 用 label manager 把平均结果转成 segmentation

支持两种场景：

- 多个推理输出文件夹 ensemble
- 多个 cross-validation 模型 ensemble

### `postprocessing/remove_connected_components.py`

作用：自动搜索“移除小连通域”是否会提升验证结果。

它的策略是：

1. 先比较“只保留最大前景连通域”是否提升
2. 再逐类/逐 region 尝试“只保留该类最大连通域”
3. 只有当指标整体更好且不伤害某类 Dice 时才采用
4. 保存：
  - `postprocessing.pkl`
  - `postprocessing.json`
  - `postprocessed/`

这就是为什么它不是简单的图像后处理脚本，而是“基于验证集自动选择后处理规则”的模块。

---

## 13. 数据集转换辅助代码

### `dataset_conversion/generate_dataset_json.py`

作用：辅助生成 `dataset.json`。

### `dataset_conversion/Dataset137_BraTS21.py`

作用：保留 BraTS21 标签转换逻辑。

包括：

- 把 BraTS segmentation 转成 nnU-Net 标签
- 再把预测结果转回 BraTS 原标签约定

它更像是历史兼容/模板文件，不是你当前 `Dataset220_BraTS2020` 主线必经入口。

---

## 14. 文档导出脚本

### `Document/export_markdown_pdfs.py`

作用：把项目里的 Markdown 文档批量导出成 PDF。

做法：

- 用 `markdown` 把 `.md` 变成 HTML
- 用 `BeautifulSoup` 调整链接
- 用 `WeasyPrint` 套 CSS 导出 PDF
- 自动生成合并手册 `BraTS_Project_Manual.pdf`

### `Document/export_with_typora.py`

作用：另一条文档导出路径。

做法：

- 自动打开 Typora
- 用 GUI 自动化脚本触发导出 PDF

这个脚本与模型无关，是文档工具层。

---

## 15. 关键非代码文件也在干什么

虽然你问的是“代码”，但这个项目里有几类非代码文件对理解主线非常关键。

### 文档层

- `README.md`
  - 顶层导览
- `TERMS_AND_THEORY.md`
  - 术语和概念解释
- `PIPELINE_EXPLANATION.md`
  - 流水线概念解释
- `BraTS_Project_Manual.md`
  - 汇总手册
- `REFERENCES.md`
  - 参考资料

### 元数据样例层

- `01_data_preparation/metadata/raw_dataset.json`
  - 转换后 raw dataset 的 `dataset.json` 样例
- `02_preprocess/metadata/dataset_fingerprint.json`
  - 指纹样例
- `02_preprocess/metadata/ProjectPlans.json`
  - 项目默认 plans 样例
- `02_preprocess/metadata/nnUNetPlans.json`
  - 兼容/对照 plans 样例
- `02_preprocess/metadata/splits_final.json`
  - 五折划分样例

### 训练结果快照

- `03_training_and_results/results/fold_0/debug.json`
  - 训练时保存的调试信息
- `03_training_and_results/results/fold_0/progress.png`
  - loss / dice 曲线图
- `03_training_and_results/results/fold_0/training_log_*.txt`
  - 训练日志文本

### 可视化输出

- `00_first_case_visualization/output/*.png/*.txt/*.json/*.csv`
  - 全都是 `visualize_first_case.py` 产物，不是手写代码

---

## 16. 这套项目真实运行时，代码是怎么串起来的

### 16.1 数据准备

```text
python run.py prepare-dataset
-> cli.cmd_prepare_dataset
-> prepare_brats2020_for_project.main
-> 生成 PROJECT_RAW/Dataset220_BraTS2020
```

这里有一个当前实现细节：

- 这条命令默认是幂等的
- 如果 `Dataset220_BraTS2020` 已经完整存在，会直接复用
- 如果你要从原始 BraTS 数据重新生成，才使用 `python run.py prepare-dataset --force`

### 16.2 规划与预处理

```text
python run.py plan-preprocess
-> cli.cmd_plan_preprocess
-> 如缺少 fingerprint，则 extract_fingerprints
   -> DatasetFingerprintExtractor.run
-> 如缺少 plans，则 plan_experiments
   -> ExperimentPlanner.plan_experiment
-> 如缺少某个 configuration 的输出，则 preprocess
   -> DefaultPreprocessor.run
```

### 16.3 训练

```text
python run.py train --fold 0
-> cli.cmd_train
-> run_training.run_training
-> get_trainer_from_args(..., trainer='SegTrainer')
-> SegTrainer(nnUNetTrainer)
-> nnUNetTrainer.run_training
```

### 16.4 验证导出

```text
训练结束
-> nnUNetTrainer.perform_actual_validation
-> nnUNetPredictor.manual_initialization
-> 滑窗推理
-> export_prediction_from_logits
-> evaluate_predictions.compute_metrics_on_folder
-> fold_x/validation/summary.json
```

### 16.5 选择最优配置

```text
python run.py find-best-config
-> evaluation.find_best_configuration
-> accumulate_cv_results
-> optional ensemble
-> determine_postprocessing
-> inference_information.json
```

默认情况下，这一步只比较当前项目默认训练组合。

但当前项目已经额外做了两层兼容：

- 如果默认组合没有可用的 `fold_x/validation` 输出，会自动回退到当前数据集下实际已经产出 validation 结果的模型
- 比较时不会再让不同模型各自使用不同的 fold 子集；系统只会使用所有候选模型共同拥有的 shared folds

如果候选模型之间连一个共享 fold 都没有，`find-best-config` 会直接报错，而不是继续输出一个不可比较的最佳模型。

如果你真的训练了多套配置，才需要再显式传入多组 `--configurations / --trainers / --plans-identifiers` 去做更广的比较。

### 16.6 正式推理

```text
python run.py predict ...
-> cli 先尝试读取 inference_information.json
-> 若存在最佳单模型，则自动解析 trainer / plans / configuration / folds
-> nnUNetPredictor.initialize_from_trained_model_folder
-> preprocess iterator
-> sliding window prediction
-> export segmentation
```

---

## 17. 哪些代码是“项目自写”，哪些更像“nnU-Net 本地分叉”

更明显偏项目自写/本地工程封装的部分：

- `run.py`
- `project_config.json`
- `brats_project/cli.py`
- `brats_project/project_layout.py`
- `01_data_preparation/scripts/prepare_brats2020_for_project.py`
- `00_first_case_visualization/visualize_first_case.py`
- `Document/*.py`
- `SegTrainer.py`

更明显偏 nnU-Net 风格核心框架的部分：

- `experiment_planning/*`
- `preprocessing/*`
- `training/*`
- `inference/*`
- `evaluation/*`
- `postprocessing/*`
- `utilities/plans_handling/*`
- `utilities/label_handling/*`
- `imageio/*`

这也是为什么你会感觉这个项目一部分很“本地化”，另一部分很“体系化”：

- 前者在解决你的目录、命令和 BraTS 数据整理问题
- 后者在复用成熟的 nnU-Net 训练框架思想

---

## 18. 如果你现在想真正读源码，建议顺序

不要从 `nnUNetTrainer.py` 第一行开始死啃。建议按下面顺序：

1. `run.py`
2. `brats_project/cli.py`
3. `project_config.json`
4. `project_layout.py`
5. `01_data_preparation/scripts/prepare_brats2020_for_project.py`
6. `experiment_planning/plan_and_preprocess_api.py`
7. `experiment_planning/dataset_fingerprint/fingerprint_extractor.py`
8. `experiment_planning/experiment_planners/default_experiment_planner.py`
9. `preprocessing/preprocessors/default_preprocessor.py`
10. `training/dataloading/data_loader.py`
11. `training/nnUNetTrainer/nnUNetTrainer.py`
12. `inference/predict_from_raw_data.py`
13. `evaluation/find_best_configuration.py`
14. `postprocessing/remove_connected_components.py`

这样读，你会先明白：

- 输入数据怎么变成训练数据
- plans 怎么决定训练配置
- trainer 最终到底在吃什么
- 推理结果又怎么回到原始空间

而不是一开始就被训练细节淹没。

---

## 19. 一句话总结整个代码库

这个 BraTS 项目不是“单个训练脚本”，而是一套完整的本地化医学图像分割流水线：

- 用 `visualize_first_case.py` 帮你理解原始 3D BraTS 数据
- 用 `prepare_brats2020_for_project.py` 把原始病例变成训练数据集
- 用 `fingerprint + planner + preprocessor` 自动生成适配该数据集的训练配置和预处理结果
- 用 `nnUNetTrainer` 完成训练、验证和 checkpoint 管理
- 用 `nnUNetPredictor` 做滑窗推理
- 用 `evaluation + ensembling + postprocessing` 自动选择最终最优推理方案

如果你后面要继续深挖，我建议你优先抓住 4 个核心文件：

- `03_training_and_results/src/brats_project/cli.py`
- `03_training_and_results/src/brats_project/experiment_planning/experiment_planners/default_experiment_planner.py`
- `03_training_and_results/src/brats_project/preprocessing/preprocessors/default_preprocessor.py`
- `03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py`

它们基本就是这套系统的“调度层、规划层、数据层、训练层”四根主梁。

---

## 20. 逐个检查代码目录后，应该怎样给它们分层

前面 1 到 19 节更偏“按流程和关键文件解释”。这一节专门回答另一个实际问题：

- 当你进入 `03_training_and_results/src/brats_project/` 逐个目录检查时，应该怎样判断哪些目录属于默认主链，哪些目录只是扩展层，哪些目录只是缓存。

先给最终分层结果：

### 20.1 入口与调度层

包括：

- `cli.py`
- `project_layout.py`
- `paths.py`
- `configuration.py`
- `run/`

这一层负责：

- 解析命令
- 定位项目和工作区路径
- 设置 `PROJECT_RAW / PROJECT_PREPROCESSED / PROJECT_RESULTS`
- 把命令行参数转成训练、推理、评估任务

这一层的特点是：

- 不直接实现分割算法
- 但它决定“当前项目默认到底跑哪条链路”

所以如果你的目标是先搞懂“现在这个项目真正怎么运行”，这里应该最先读。

### 20.2 主算法流水线层

包括：

- `experiment_planning/`
- `preprocessing/`
- `imageio/`
- `utilities/`
- `training/`
- `inference/`
- `evaluation/`
- `ensembling/`
- `postprocessing/`

这是整个 nnU-Net 风格主链真正发生的地方：

1. `experiment_planning/`
   - 根据数据统计生成 plans
2. `preprocessing/`
   - 按 plans 执行裁剪、重采样、归一化和序列化
3. `training/`
   - 完成 dataloader、augmentation、loss 和 trainer 主循环
4. `inference/`
   - 完成滑窗推理和预测导出
5. `evaluation/ensembling/postprocessing/`
   - 完成指标计算、模型组合和预测后处理

判断一个目录是不是主链目录，最简单的标准是：

- 默认命令会不会直接或间接调用到它
- 它会不会影响 `ProjectPlans.json`、训练行为或推理结果

按这个标准，上面这些目录都属于当前项目的核心层。

### 20.3 扩展与兼容层

包括：

- `dataset_conversion/`
- `training/nnUNetTrainer/variants/`
- `training/nnUNetTrainer/primus/`
- `experiment_planning/plans_for_pretraining/`

这些目录的共同特点是：

- 它们不是当前默认 BraTS2020 主链的关键路径
- 但它们也不是可以直接视为垃圾的代码

更具体地说：

- `dataset_conversion/`
  - 保留通用数据集接入和其它数据集脚本
- `variants/`
  - 保留大量 trainer 变体，覆盖增强、loss、lr、optimizer、sampling、训练长度、benchmark、competition 等方向
- `primus/`
  - 更像某组特定实验或作者扩展
- `plans_for_pretraining/`
  - 面向 plans 迁移和预训练场景

这一层最容易被误删，因为：

- “当前默认命令没走到”不等于“永远无用”
- “不是 BraTS2020 主路径”也不等于“应该删掉”

更合理的处理方式是：

- 先在文档中显式标成扩展层
- 先把阅读顺序排到主链之后
- 只有确认没有导入、没有命令入口、没有实验依赖时，再考虑瘦身

### 20.4 缓存与非源码层

包括：

- `__pycache__/`

这一层的判断最明确：

- 不是源码
- 不参与代码理解
- 只要里面是 `.pyc`，就可以作为编译缓存安全清理

### 20.5 这次检查后，什么能删，什么不该随便删

安全可删：

- `__pycache__/` 下的 `.pyc`

暂不建议直接删：

- `dataset_conversion/` 中非当前数据集脚本
- `training/nnUNetTrainer/variants/`
- `training/nnUNetTrainer/primus/`
- `experiment_planning/plans_for_pretraining/`

原因不是它们现在都在主链里，而是：

- 它们仍然属于框架保留能力
- 没有进一步证据时，删掉的风险高于保留

### 20.6 如果你现在要继续“逐目录读源码”，顺序应该怎样排

推荐顺序：

1. `cli.py`
2. `project_layout.py`
3. `run/run_training.py`
4. `experiment_planning/`
5. `preprocessing/`
6. `training/dataloading/`
7. `training/nnUNetTrainer/SegTrainer.py`
8. `training/nnUNetTrainer/nnUNetTrainer.py`
9. `inference/`
10. `evaluation/`
11. `postprocessing/`
12. `utilities/`
13. `training/nnUNetTrainer/variants/`
14. `dataset_conversion/`

这样读的最大好处是：

- 你先把默认主链读通
- 再回头看扩展层
- 不会一开始就陷进大量当前项目根本没走到的实验代码

如果你想看更聚焦的目录审计版说明，还可以直接看：

- `03_training_and_results/docs/code_directory_audit.md`

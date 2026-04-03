# Preprocess Guide

这份 guide 的目标不是重复命令，而是解释：

- preprocess 阶段真正决定了什么
- 为什么它是训练前最值得读懂的一层
- 当前项目里哪些问题看起来像训练问题，根因却常常在这里

---

## 1. 先记住这一阶段的调用链

从项目入口看，调用链大致是：

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [plan_and_preprocess_api.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/plan_and_preprocess_api.py)
4. [fingerprint_extractor.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/dataset_fingerprint/fingerprint_extractor.py)
5. [default_experiment_planner.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/experiment_planners/default_experiment_planner.py)
6. [default_preprocessor.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/preprocessing/preprocessors/default_preprocessor.py)

这个顺序对应的不是“文件先后”，而是：

- 先理解数据
- 再生成方案
- 再执行方案

---

## 2. 这一阶段到底在决定什么

最容易被低估的是：preprocess 阶段并不是给训练做“前置清洗”，而是在真正决定训练器看到的数据表示。

它会固定下来：

- 轴顺序
- 裁剪范围
- 统一后的 spacing
- 训练 patch 所在的尺度语义
- 强度归一化策略
- 前景采样辅助信息
- 预处理后存储格式

你可以把它理解成：

- “训练数据的最后定稿层”

---

## 3. fingerprint 应该怎么看

### 不要只把它当统计汇总

`dataset_fingerprint.json` 不是留档用的文件，而是 planner 的输入。

它的价值在于：

- 它把 planner 的“依据”显式保存了下来

### 应重点看的字段

- `spacings`
  - 原始 spacing 分布
- `shapes_after_crop`
  - 非零裁剪后各病例形状
- `median_relative_size_after_cropping`
  - 裁剪后体积相对比例
- `foreground_intensity_properties_per_channel`
  - 前景强度统计

### 为什么先看它比先看 plans 更好

因为：

- fingerprint 是 planner 的前提
- plans 是 planner 的结论

如果你只看结论，不看依据，很容易把很多设置误以为是“写死的经验参数”。

---

## 4. plans 应该怎么看

### `ProjectPlans.json` 的本质

它不是手调参数清单，而是：

- 当前数据集
- 当前 planner
- 当前显存目标

共同产出的正式决议。

### 当前 BraTS 项目里最值得关注的字段

- `spacing`
- `patch_size`
- `batch_size`
- `normalization_schemes`
- `use_mask_for_norm`
- `resampling_fn_data`
- `resampling_fn_seg`
- `architecture`

### 为什么 patch 和 batch 值很重要

因为它们几乎决定了：

- 每次训练看到多少上下文
- GPU 占用方式
- 网络每次更新的数据量

在 3D 医学图像任务中，这些值通常不是凭经验随便拍出来的，而是显存约束和数据尺度共同作用的结果。

---

## 5. preprocessor 对单病例的真实处理顺序

当前默认 preprocessor 的主线逻辑是：

1. 读取多模态图像和标签
2. 根据 plans 执行 transpose
3. 裁到非零区域
4. 记录裁剪前后 shape
5. 计算目标 shape
6. 执行归一化
7. 执行重采样
8. 记录前景位置 `class_locations`
9. 保存成训练期快速读取格式

这里有两个非常值得记住的点。

### 点 1：归一化发生在重采样前

这是有意设计的，不是实现细节。

原因是：

- 对 BraTS 这种脑 MRI，非零 mask 和前景统计非常关键
- 先重采样再算归一化会让 mask 与强度统计关系更复杂

### 点 2：会保存 `class_locations`

这一步对后面的 dataloader 非常重要。

因为 dataloader 要做前景过采样，必须知道：

- 哪些位置包含前景

这些信息就是 preprocess 阶段先算好的。

---

## 6. 当前 BraTS 数据为什么主要盯住 `3d_fullres`

对你当前项目来说，默认主线是：

- trainer: `SegTrainer`
- plans: `ProjectPlans`
- configuration: `3d_fullres`

这背后意味着：

- planner 已经认为 `3d_fullres` 是当前最直接、最自然的主训练配置
- 后面训练文档也围绕这条主线组织

这不代表别的 configuration 毫无意义，而是：

- 你当前项目的默认工程约定已经收敛到 `3d_fullres`

---

## 7. 最常见的误判

### 误判 1：preprocess 出错，训练一定也会直接报错

不一定。

更麻烦的情况是 preprocess 没完全错，但表示层已经不合理，于是后面训练能跑，却表现很怪。

### 误判 2：看到 plans 就应该先手调 patch size

通常不应该。

先问：

- planner 为什么给出这个值
- 当前 fingerprint 是否支持你想改的方向

### 误判 3：训练器读的是 raw NIfTI

不是。

训练器主要读的是 preprocess 后的数据。

### 误判 4：split 是训练阶段才重要

不对。

它在 preprocess 阶段就已经成为后续 fold 语义的正式来源。

---

## 8. 真正值得检查的文件顺序

如果你在排查 preprocess 阶段，建议按下面顺序：

1. `dataset.json`
2. `dataset_fingerprint.json`
3. `ProjectPlans.json`
4. `splits_final.json`
5. preprocess 输出目录中的 case 文件

这样查的好处是：

- 先验证任务定义
- 再验证 planner 依据
- 再验证 planner 结论
- 再验证 fold 语义
- 最后看具体预处理数据

---

## 9. 这一阶段和训练阶段的接口

训练阶段真正依赖这里提供的，不只是“预处理后的数据目录”。

它还依赖：

- `ProjectPlans.json`
- `dataset.json`
- `splits_final.json`
- `class_locations`
- 统一后的 `data_identifier`

所以训练的主入口虽然在 `run_training.py`，但训练的输入语义其实是在这一层被定死的。

---

## 10. 最后建议

如果你准备继续深读项目，不建议直接跳去 trainer。

更好的顺序是：

1. 先把 `dataset_fingerprint.json` 和 `ProjectPlans.json` 对着看懂
2. 再去读 `default_preprocessor.py`
3. 然后再看 `data_loader.py`
4. 最后再看 `nnUNetTrainer.py`

这样你能清楚知道 trainer 吃进去的到底是什么，而不是只看到训练循环本身。

# Pipeline Explanation

这份文档不是代码索引，而是“按真实执行顺序把整个项目串成一条链”。

它回答的问题不是“某个函数在干嘛”，而是：

- 一条命令从哪里进入
- 它的输入是什么
- 它会产出什么
- 下一个阶段为什么要依赖这些产物
- 如果这一层出错，后面通常会表现成什么问题

---

## 0. 先把整条链记住

```text
原始 BraTS 病例
  -> 第一个病例可视化与对齐检查
  -> 原始目录转换成 Dataset220_BraTS2020
  -> fingerprint 提取
  -> experiment planning
  -> preprocess
  -> train / validation
  -> 汇总 cross-validation
  -> best configuration selection
  -> inference
  -> postprocessing
  -> evaluation
```

这个顺序不是文档习惯，而是项目真实依赖关系。

---

## 阶段 1：先建立数据直觉

### 对应命令

```bash
python BraTS/run.py visualize-first-case
```

### 对应代码

- [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
- [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
- [visualize_first_case.py](/home/Creeken/Desktop/machine-learning-test/BraTS/00_first_case_visualization/visualize_first_case.py)

### 输入

- 原始 BraTS 根目录
- 一个按 BraTS 风格组织的病例文件夹集合

### 输出

- `case_summary.txt/json`
- `seg_labels_summary.txt`
- `intensity_stats.csv`
- 多张 PNG 可视化图

### 这一步理论上在解决什么

它在解决“后面所有代码默认成立，但你肉眼还没确认”的东西：

- 多模态 MRI 是否真的对齐
- `seg` 是否真的是同空间 3D label volume
- BraTS 不同模态在病灶显示上到底有什么差异
- 标签 `0/1/2/4` 的空间分布大概怎样

### 如果这一层有问题，后面常见表现

- overlay 看起来位置不对
- 数据转换后 geometry 校验失败
- 训练时标签和图像对不上
- 你在后面看 trainer 和 predictor 时完全没有空间感

---

## 阶段 2：把原始病例变成训练目录

### 对应命令

```bash
python BraTS/run.py prepare-dataset
```

这条命令现在默认支持重复执行：

- 第一次运行会真正生成 `Dataset220_BraTS2020`
- 之后如果目标目录已经完整存在，会直接复用
- 如果你想强制重建，使用 `python BraTS/run.py prepare-dataset --force`

### 对应代码

- [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
- [prepare_brats2020_for_project.py](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/scripts/prepare_brats2020_for_project.py)

### 输入

- 原始病例目录
- 每个病例包含：
  - `t1`
  - `t1ce`
  - `t2`
  - `flair`
  - `seg`

### 输出

- `PROJECT_RAW/Dataset220_BraTS2020/imagesTr`
- `PROJECT_RAW/Dataset220_BraTS2020/labelsTr`
- `PROJECT_RAW/Dataset220_BraTS2020/dataset.json`

### 这一步具体做什么

1. 扫描合法病例
2. 校验各模态和标签几何一致
3. 把模态改写为统一通道编号：
   - `0000 = T1`
   - `0001 = T1ce`
   - `0002 = T2`
   - `0003 = Flair`
4. 把 BraTS 原始标签映射成项目训练标签：
   - `0 -> 0`
   - `2 -> 1`
   - `1 -> 2`
   - `4 -> 3`
5. 生成带 region 定义的 `dataset.json`

### 这一步为什么是“数据契约层”

后面的所有模块都不再直接理解“原始 BraTS 文件名和原始标签编码”，而是理解：

- 统一通道编号
- 统一 labels/regions 定义
- 统一的 `dataset.json`

所以这一步做错，后面不会立即都崩，但会在更隐蔽的地方表现为：

- 标签语义错位
- 通道顺序错位
- 训练结果异常
- 推理语义错误

---

## 阶段 3：提取 fingerprint

### 对应命令

```bash
python BraTS/run.py plan-preprocess
```

这条命令包含多个子步骤，fingerprint 是第一步。
但当前项目里，它默认会优先复用已有 `dataset_fingerprint.json`、`ProjectPlans.json` 和已有 preprocess 输出，只在缺失时补做。

如果你确实要强制重算，才使用：

```bash
python BraTS/run.py plan-preprocess --recompute-fingerprint
python BraTS/run.py plan-preprocess --recompute-plans
python BraTS/run.py plan-preprocess --force-preprocess
python BraTS/run.py plan-preprocess --clean
```

### 对应代码

- [plan_and_preprocess_api.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/plan_and_preprocess_api.py)
- [fingerprint_extractor.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/dataset_fingerprint/fingerprint_extractor.py)

### 输入

- `PROJECT_RAW/Dataset220_BraTS2020`
- `dataset.json`

### 输出

- `PROJECT_PREPROCESSED/Dataset220_BraTS2020/dataset_fingerprint.json`

### fingerprint 在记录什么

它不是一份“随手统计”文件，而是 planner 的依据。

它主要记录：

- 各病例 spacing
- 裁剪后 shape
- 裁剪后占原体积比例
- 前景强度统计

### 为什么它重要

planner 不直接看原始数据逐例猜参数，而是先基于 fingerprint 做全数据集层面的判断：

- target spacing 应该取多少
- 数据有多各向异性
- patch 该有多大
- batch size 能有多少
- 是否需要低分辨率配置

---

## 阶段 4：生成 plans

### 对应命令

仍然是：

```bash
python BraTS/run.py plan-preprocess --clean
```

### 对应代码

- [default_experiment_planner.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/experiment_planners/default_experiment_planner.py)
- [plans_handler.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/utilities/plans_handling/plans_handler.py)

### 输入

- `dataset_fingerprint.json`
- `dataset.json`

### 输出

- `ProjectPlans.json`
- 可选的 `nnUNetPlans.json`

### plans 到底在决定什么

它在做“把数据特点翻译成训练方案”的工作。

它会决定：

- transpose 方向
- fullres target spacing
- 2D / 3D 配置是否存在
- 是否需要 `3d_lowres`
- patch size
- batch size
- 网络拓扑参数
- 归一化方案
- 重采样函数

### 这一步的真实意义

这一步是整个 nnU-Net 思想的核心枢纽：

- fingerprint 是“数据画像”
- plans 是“训练配方”

对你当前 BraTS 数据，最终主线通常会收敛到：

- `ProjectPlans`
- `3d_fullres`

---

## 阶段 5：执行 preprocess

### 对应命令

同样还是：

```bash
python BraTS/run.py plan-preprocess --clean
```

### 对应代码

- [default_preprocessor.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/preprocessing/preprocessors/default_preprocessor.py)
- [cropping.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/preprocessing/cropping/cropping.py)
- [default_resampling.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/preprocessing/resampling/default_resampling.py)

### 输入

- `PROJECT_RAW/Dataset220_BraTS2020`
- `ProjectPlans.json`

### 输出

- `PROJECT_PREPROCESSED/Dataset220_BraTS2020/<data_identifier>/`
- `splits_final.json`
- `gt_segmentations/`

### preprocess 对单个病例做什么

顺序大致是：

1. 读多模态图像和标签
2. 根据 plans 做轴转置
3. 裁到非零区域
4. 计算目标 shape
5. 归一化
6. 重采样
7. 采样前景位置并写入 `class_locations`
8. 保存成训练期快速读取格式

### 为什么这一步决定训练器“真正吃到什么”

trainer 不直接读 `imagesTr/labelsTr` 的 NIfTI。

trainer 读的是 preprocess 后的数组/压缩块文件以及对应 properties，所以：

- 数据的真实 patch 尺度
- 深度监督 target 结构
- 前景过采样依据

都在这一层被固定下来。

---

## 阶段 6：训练与在线验证

### 对应命令

```bash
python BraTS/run.py train --fold 0
python BraTS/run.py train-all --npz
```

### 对应代码

- [run_training.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/run/run_training.py)
- [SegTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/SegTrainer.py)
- [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)
- [data_loader.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/dataloading/data_loader.py)

### 输入

- preprocess 后的数据目录
- `ProjectPlans.json`
- `dataset.json`
- `splits_final.json`

### 输出

- checkpoint
- log
- `debug.json`
- `progress.png`
- `validation/` 预测结果
- 如果带 `--npz`，还会有 validation 概率输出

### 训练主链在做什么

1. 从 trainer 名字找到 trainer 类
2. 读入 plans 和 dataset.json
3. 构造 `LabelManager`
4. 构造网络、优化器、调度器、loss
5. 构造 dataloader 和增强流水线
6. 按 fold 对 train/val 病例做划分
7. 执行 epoch 循环
8. 写 checkpoint 和日志
9. 训练完成后做真实 validation 推理并导出 summary

### 为什么建议先跑 `fold_0`

因为训练阶段最先要排除的是工程链路问题，而不是模型表现问题：

- trainer 类找不到
- 路径不对
- plans 名称不一致
- GPU 不可用
- 结果目录无权限

这些问题在 `fold_0` 就能暴露。

---

## 阶段 7：把五折结果汇总成可比较对象

### 对应命令

通常由：

```bash
python BraTS/run.py find-best-config
```

间接触发。

### 对应代码

- [accumulate_cv_results.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/accumulate_cv_results.py)

### 输入

- 各 fold 的 `validation` 输出

### 输出

- `crossval_results_folds_xxx/`
- 对应的 `summary.json`

### 这一步为什么必要

因为单个 fold 只反映某一次划分下的验证结果。

而后面要回答的问题是：

- 当前默认训练组合整体表现怎样
- 如果你额外训练了别的 configuration，哪个整体最好
- 是否值得 ensemble
- 哪个模型值得拿去正式推理

这些问题都需要多折汇总结果，而不是单折结果。

---

## 阶段 8：选择最佳配置

### 对应命令

```bash
python BraTS/run.py find-best-config
```

### 对应代码

- [find_best_configuration.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/find_best_configuration.py)
- [ensemble.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/ensembling/ensemble.py)
- [remove_connected_components.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/postprocessing/remove_connected_components.py)

### 输入

- 默认是当前项目默认训练组合的 cross-validation 汇总结果
- 如果显式传了多个 `--configurations/--trainers/--plans-identifiers`，则比较这些组合的汇总结果
- 如允许，还包括用于 ensemble 的 `.npz` 概率输出

### 输出

- `inference_information.json`
- `inference_instructions.txt`
- 最佳模型或 ensemble 的后处理规则

### 这一步具体做什么

1. 检查哪些 trained model 实际可用
2. 汇总每个单模型的 crossval 结果
3. 如果允许，构造两两 ensemble 并重新评估
4. 选出最优者
5. 自动寻找最优后处理规则
6. 生成正式推理说明

### 为什么它不是“可选装饰”

因为这一步不仅仅是打印一个 best score。

它实际上在决定：

- 后面正式推理到底用哪个 configuration
- 是否需要 ensemble
- 推理后需要应用哪套 postprocessing

---

## 阶段 9：正式推理

### 对应命令

```bash
python BraTS/run.py predict
```

### 对应代码

- [predict_from_raw_data.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/predict_from_raw_data.py)
- [data_iterators.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/data_iterators.py)
- [sliding_window_prediction.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/sliding_window_prediction.py)
- [export_prediction.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/export_prediction.py)

### 输入

- 正式推理输入目录
- 训练好的模型目录
- `plans.json`
- `dataset.json`
- 若为 ensemble，则多个模型结果

### 输出

- `.nii.gz` segmentation
- 推理参数快照
- 复制过来的 `dataset.json/plans.json`
- 可能还有 `.npz` 概率图

### 推理真正做什么

1. 从训练输出目录恢复模型、plans、dataset.json
2. 如果项目默认输入目录为空，CLI 会先从训练集随机抽样少量病例，把它们复制成临时验证输入
3. 对输入病例执行与训练一致的 preprocess
4. 执行滑窗预测
5. 对多 fold 权重求平均
6. 把 logits 逆重采样、逆裁剪、逆 transpose 回原始空间
7. 写出正式 segmentation

### 正式推理最常见的错误来源

- 输入通道顺序和训练不一致
- 用错 trainer/plans/configuration
- 模型目录不完整
- ensemble 需要的概率文件没有保留

当前项目为了更方便做链路调试，`python BraTS/run.py predict` 还有一层额外默认逻辑：

- 如果 `BraTS/04_inference_and_evaluation/input` 为空
- CLI 会自动从 `PROJECT_RAW/Dataset220_BraTS2020/imagesTr` 随机抽样 `8` 个训练病例
- 把它们复制到输入目录中，作为临时验证集
- 抽样记录会写到 `BraTS/04_inference_and_evaluation/input/sample_selection.json`

---

## 阶段 10：后处理

### 对应命令

通常由 `find-best-config` 自动决定规则，之后可用：

```bash
brats_apply_postprocessing ...
```

### 对应代码

- [remove_connected_components.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/postprocessing/remove_connected_components.py)

### 输入

- 原始预测结果
- GT 或验证参考结果

### 输出

- `postprocessing.pkl`
- `postprocessing.json`
- `postprocessed/`

### 它不是在做什么

它不是“固定地删小连通域”。

它是在验证集上测试：

- 去掉非最大前景连通域是否提升
- 逐类去掉小连通域是否提升

只有当效果确实更好且不伤害关键指标时，规则才会被采用。

---

## 阶段 11：正式评估

### 对应命令

```bash
python BraTS/run.py evaluate
```

### 对应代码

- [evaluate_predictions.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/evaluate_predictions.py)

### 输入

- GT segmentation
- prediction segmentation

### 输出

- `summary.json`
- 按 case / 按 label 或 region 的 Dice、IoU、TP/FP/FN/TN

### 这一步回答的不是单一分数，而是结构化问题

- 每个病例表现怎样
- 每个 region 表现怎样
- 前景总体均值怎样

所以它既是最终评价，也常是回头定位问题的入口。

---

## 12. 每一层失败通常会往哪里表现

### 数据直觉层失败

常表现为：

- 你看不懂后面所有产物
- overlay 不知道正常不正常

### 数据转换层失败

常表现为：

- 通道顺序错
- 标签语义错
- 训练结果很奇怪但不一定直接报错

### fingerprint / plans 层失败

常表现为：

- patch size/batch size 不合理
- preprocess 与训练资源使用异常

### preprocess 层失败

常表现为：

- dataloader 问题
- patch 采样怪异
- 训练和推理的空间还原不一致

### 训练层失败

常表现为：

- 日志/ckpt 不生成
- validation 失败
- pseudo dice 和最终验证不一致

### 推理层失败

常表现为：

- 输出 shape 或 geometry 不对
- 输入目录明明能读，结果却语义错位

### 后处理/评估层失败

常表现为：

- best config 选择异常
- postprocessing 规则无意义
- summary.json 看起来和训练直觉不一致

---

## 13. 推荐你怎样利用这条 pipeline

如果你是第一次真正读懂这个项目，不要试图一次性把所有模块吃掉。

建议顺序：

1. 先把阶段 1 和阶段 2 搞清楚
2. 再看阶段 3 到阶段 5
3. 然后再看训练主链
4. 最后看 best config、推理、后处理和评估

也就是：

- 先理解输入
- 再理解中间表示
- 再理解优化过程
- 最后理解结果选择与交付

这样这条 pipeline 才会真正变成一张稳定的脑内地图，而不是一堆文件名。

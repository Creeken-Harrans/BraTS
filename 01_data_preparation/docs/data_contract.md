# Data Contract

这份文档定义的是当前 BraTS 项目最底层的数据契约。

它不是“建议格式”，而是后面的：

- fingerprint
- planner
- preprocessor
- trainer
- predictor

能够稳定工作的前提。

一句话概括：

- 只要训练目录违反这份契约，后面的流程要么直接报错，要么更糟糕地在更晚阶段以隐蔽方式出错。

---

## 1. 数据契约分哪几层

当前项目的数据契约可以分成五层：

1. 目录契约
2. 文件命名契约
3. 通道语义契约
4. 标签与 region 语义契约
5. 空间几何契约

只有这五层同时成立，后面的计划生成、训练、推理、评估才有可靠输入。

---

## 2. 目录契约

训练目录必须命名为：

- `Dataset220_BraTS2020`

并且至少包含：

- `imagesTr/`
- `labelsTr/`
- `dataset.json`

这是最基本的 nnU-Net 风格目录契约。

如果缺其中任意一项，后面的主链都会失去稳定输入。

---

## 3. 文件命名契约

### 图像文件命名

每个 case 必须有 4 个模态文件，命名形式如下：

- `{case_id}_0000.nii.gz`
- `{case_id}_0001.nii.gz`
- `{case_id}_0002.nii.gz`
- `{case_id}_0003.nii.gz`

### 标签文件命名

- `{case_id}.nii.gz`

### 为什么文件名这么重要

因为后面的工具不是靠“医学常识猜模态”，而是靠：

- case identifier
- 通道编号

去组织输入。

也就是说，如果你把模态顺序放错，程序不一定马上报错，但会把错误通道送进模型。

---

## 4. case identifier 契约

同一个病例的四个模态和一个标签必须共享同一个 `case_id`。

例如：

- `BraTS20_Training_001_0000.nii.gz`
- `BraTS20_Training_001_0001.nii.gz`
- `BraTS20_Training_001_0002.nii.gz`
- `BraTS20_Training_001_0003.nii.gz`
- `BraTS20_Training_001.nii.gz`

这里：

- `BraTS20_Training_001`

就是 case identifier。

它的重要性在于：

- dataloader、split、predictor、evaluation 都是围绕 case identifier 组织的

---

## 5. 通道语义契约

当前项目要求：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

这条契约必须在：

- 训练目录
- 推理输入目录

两个阶段都严格成立。

### 不能出现的错误

- 某些病例把 `T2` 写进 `0001`
- 推理阶段把 `Flair` 和 `T1ce` 对调
- 缺一个模态但仍然保留 4 通道命名假象

### 为什么这是硬约束

因为模型只知道：

- 第 0 通道是什么
- 第 1 通道是什么

它不知道你的文件从医学上“原本想表示什么”。

---

## 6. 标签语义契约

### 原始 BraTS 标签

- `0 = background`
- `1 = NCR/NET`
- `2 = ED`
- `4 = ET`

### 项目训练标签

当前项目内部训练表示会重编码为：

- `0 -> 0`
- `2 -> 1`
- `1 -> 2`
- `4 -> 3`

也就是说，训练目录中的：

- `1 / 2 / 3`

已经不是原始 BraTS 的数字本身，而是项目内部训练语义。

### 为什么要重编码

不是因为“连续整数好看”，而是因为：

- 后面的 region-based training 需要一个更适合内部组合和表达的标签体系

---

## 7. region 契约

当前项目不是把单个 label 直接当作最终优化目标，而是按 BraTS region 组织训练目标。

`dataset.json` 中的关键定义是：

- `whole_tumor = [1, 2, 3]`
- `tumor_core = [2, 3]`
- `enhancing_tumor = [3]`

并且：

- `regions_class_order = [1, 2, 3]`

### 为什么 `regions_class_order` 很关键

因为 region 输出在还原 segmentation 时是有顺序语义的。

如果这里顺序错了，可能发生：

- 代码不报错
- 模型也能训练
- 但最终不同输出通道的语义被错误解释

这是一类非常危险的“静默错误”。

---

## 8. `dataset.json` 契约

在当前项目里，`dataset.json` 不只是目录说明，而是任务定义文件。

它至少要稳定提供：

- `channel_names`
- `labels`
- `regions_class_order`
- `numTraining`
- `file_ending`

### 它被谁消费

- planner
- preprocessor
- label manager
- trainer
- predictor
- evaluation

所以它不是“给人看的 JSON”，而是被整个系统实际读取的任务入口。

---

## 9. 空间几何契约

同一个病例内部，4 个模态与标签必须满足：

- shape 一致
- spacing 一致
- origin 一致
- direction 或 affine 一致

### 为什么这是必须项

因为：

- 多模态会直接按 channel 堆叠
- segmentation 默认逐 voxel 对齐
- overlay 和逆导出都依赖 geometry 一致

### 这条契约如果被破坏，会有什么后果

可能表现为：

- overlay 明显错位
- preprocess 中 geometry 检查失败
- 训练看起来能跑，但标签监督失真
- 推理导回原空间后结果位置不对

---

## 10. 推理输入必须复用同一套契约

不要误以为这份契约只对训练目录有效。

它同样约束正式推理输入：

- 每个 case 依然要 4 通道
- 通道顺序依然要 `0000~0003`
- 模态语义依然不能变

也就是说：

- 训练期和推理期的数据命名契约必须一致

---

## 11. 如何检查一个目录是否满足这份契约

建议按下面顺序检查：

1. 目录是否存在 `imagesTr/labelsTr/dataset.json`
2. `imagesTr` 中每个 case 是否正好对应 4 个 `_000x`
3. `labelsTr` 中是否存在同 `case_id` 的标签文件
4. `dataset.json` 中通道定义和标签定义是否正确
5. 随机抽一例确认 geometry 是否一致

如果你已经跑过：

```bash
python BraTS/run.py visualize-first-case
```

那么你至少已经对 geometry 和模态角色有了一次可视化确认。

---

## 12. 常见错误清单

### 错误 1：通道顺序错，但文件名看起来都合法

这是最危险的一类错误。

模型会正常读取，但语义全错。

### 错误 2：标签仍然保留原始 `0/1/2/4`

如果项目内部训练目录没有完成重编码，后面的 region 定义和 label handling 会被破坏。

### 错误 3：`dataset.json` 被随手改动

尤其危险的是：

- 调换 label 字段顺序
- 改坏 `regions_class_order`
- 改错 channel names

### 错误 4：图像和标签 geometry 不一致

这是最基础但也最致命的错误之一。

### 错误 5：训练和推理的输入契约不一致

训练按一套通道顺序，推理又按另一套顺序喂模型，通常不会立刻崩，但结果会严重失真。

---

## 13. 这份契约和后续阶段的关系

### 对 `02_preprocess`

它决定 planner 和 preprocessor 看到的任务定义是否可靠。

### 对 `03_training_and_results`

它决定 trainer 看到的输入通道数和 label/region 语义是否正确。

### 对 `04_inference_and_evaluation`

它决定 predictor 能否把推理输入解释成与训练一致的模态排列。

---

## 14. 结论

这份 data contract 不是辅助文档，而是整个项目最底层的稳定前提。

如果你后面碰到：

- 训练结果怪
- 推理结果怪
- labels 好像不对
- overlay 好像不对

第一反应应该不是先去改 trainer，而是先回来重新核对这里的契约是否被破坏。

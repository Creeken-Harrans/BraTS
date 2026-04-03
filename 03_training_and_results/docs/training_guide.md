# Training Guide

命令位置说明：
- 本文默认假设你当前目录就是 `BraTS` 项目根目录，因此命令示例写成 `python run.py ...`。
- 如果你当前在上一级目录 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径默认也都相对于 `BraTS` 项目根目录。

这份 guide 重点解释四件事：

1. 训练入口怎样真正走到 trainer
2. trainer 初始化时到底依赖哪些文件和对象
3. fold、validation、checkpoint 在当前项目中分别扮演什么角色
4. 训练结果目录应该怎样被解释

---

## 1. 训练入口的真实调用链

真实调用链大致是：

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [run_training.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/run/run_training.py)
4. [SegTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/SegTrainer.py)
5. [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)

这个顺序的重要性在于，它把两层东西分开了：

- 项目层
  - 负责路径、默认参数、命令组织
- 训练层
  - 负责真正优化模型参数

所以排错时要先判断：

- 这是入口/配置问题
  还是
- trainer 内部问题

---

## 2. `run_training.py` 到底做了什么

这个文件不是训练循环本身，而是 trainer 启动器。

它做的核心工作是：

1. 根据名字找到 trainer 类
2. 读取数据集名、configuration、plans
3. 加载 `ProjectPlans.json` 和 `dataset.json`
4. 构造 trainer 实例
5. 判断是：
   - 新训练
   - 继续训练
   - 只做 validation
   - 载入预训练权重
6. 决定单卡还是 DDP 多卡

当前项目里，默认训练入口已经统一成下面这个策略：

- 发现当前 fold 存在旧训练状态时，自动续训
- 找不到可恢复 checkpoint 时，自动从头开始
- 自动续训的 checkpoint 优先级是：`checkpoint_final` -> `checkpoint_latest` -> `checkpoint_best`
- 只有显式传 `--restart-training`，才会忽略已有 checkpoint

也就是说，它真正回答的是：

- “要启动哪个 trainer，以什么上下文启动”

---

## 3. trainer 初始化依赖哪些文件

默认主线下，训练至少依赖：

- `PROJECT_PREPROCESSED/Dataset220_BraTS2020/ProjectPlans.json`
- `PROJECT_PREPROCESSED/Dataset220_BraTS2020/dataset.json`
- `PROJECT_PREPROCESSED/Dataset220_BraTS2020/splits_final.json`
- 对应 configuration 的 preprocess 输出目录

这些文件分别扮演的角色是：

### `ProjectPlans.json`

负责告诉 trainer：

- 当前 configuration 的 patch size
- batch size
- normalization
- resampling
- architecture 参数

### `dataset.json`

负责告诉 trainer：

- 输入通道数
- label / region 定义
- `regions_class_order`

### `splits_final.json`

负责告诉 trainer：

- 当前 fold 的 train/val case 划分

### preprocess 输出目录

负责提供 trainer 真正读取的训练数据表示。

---

## 4. `SegTrainer + ProjectPlans` 为什么是默认组合

当前项目把默认入口统一收敛成：

- `SegTrainer`
- `ProjectPlans`

这样做的价值不是“名字更好看”，而是：

- 避免上游项目大量命名在本地项目里来回切换
- 让 CLI、文档、目录命名和默认训练路径保持稳定

对你当前这份 BraTS 数据来说，这个组合已经成为：

- 项目默认工程语义

---

## 5. `fold` 在训练阶段的真实含义

`fold` 不是单纯编号，它代表：

- 一份从 `splits_final.json` 读取出来的 train/val 划分

也就是说：

- `fold_0`
  - 是第 0 组 train/val 划分
- `fold_1`
  - 是第 1 组划分

所以：

- 跑 `fold_0`
  - 不是随便先试一下
  - 而是先用第一组交叉验证划分验证训练链

---

## 6. trainer 主体里最重要的几个阶段

在 `nnUNetTrainer.py` 中，最重要的生命周期可以这样理解：

### 初始化阶段

负责：

- 设置 device
- 构造 plans manager / configuration manager / label manager
- 构造网络、优化器、scheduler、loss

### dataloader 准备阶段

负责：

- 根据 preprocess 产物构造 dataset
- 根据 `class_locations` 做前景过采样
- 配置训练/验证 transform

### 训练循环阶段

负责：

- train step
- validation step
- 记录 train loss / val loss / pseudo dice

### checkpoint 与日志阶段

负责：

- 每个 epoch 覆盖更新 latest
- 记录 best
- 收尾保存 final
- 恢复训练时只从磁盘上的 checkpoint 文件恢复，不再暴露“传入内存 checkpoint dict”这种未实现接口

### 真实 validation 阶段

负责：

- 训练结束后对 held-out fold 做正式滑窗推理
- 导出 `validation` 结果
- 生成 `summary.json`

这一层很关键，因为：

- 训练过程中的在线验证指标
  不等于
- 最终 validation 文件夹里的真实推理结果

---

## 7. `--npz` 为什么最好在完整五折里保留

训练指南里最常被忽略的一点是：

- validation 概率输出不是多余附件

它直接服务于：

- cross-validation 比较
- ensembling
- `find-best-config`

并且当前项目的 `find-best-config` 只会在所有候选模型共享的 validation folds 上做比较。
如果你让不同 configuration 只各自跑了互不重叠的 folds，后面不会再给出“最佳模型”，而是会直接报错。

如果你后面打算做正式的 best configuration selection，五折训练最好带上：

```bash
python run.py train-all --npz
```

---

## 8. 结果目录应该怎么读

### `debug.json`

适合回答：

- 实际用了什么 plans
- 输出目录是什么
- 当前配置和设备是什么
- dataloader/transform 是什么

### `training_log_*.txt`

适合回答：

- 训练是否正常推进
- epoch 在哪里
- checkpoint 是否保存
- 这次到底是从头训练，还是从哪个 checkpoint 继续
- checkpoint 恢复后当前接续到哪个 epoch
- 恢复来源是哪个 checkpoint 文件路径

### `progress.png`

适合回答：

- train loss、val loss、pseudo dice 的趋势
- 当前 fold 的训练曲线是否持续正常更新

但它不能回答：

- 真正最终哪个配置最好
- 哪个结果最适合正式推理

### `validation/summary.json`

这是后面：

- best configuration selection
- crossval 汇总
- 后处理比较

真正要依赖的结果之一。

---

## 9. 当前目录里的样例结果怎么用

`results/fold_0/` 更像：

- 训练输出结构样例

而不是：

- 生产环境模型资产

使用它的方式应该是：

- 熟悉文件类别
- 熟悉日志格式
- 熟悉后面新训练时去哪里找信息

而不是把它当成最终模型复用入口。

---

## 10. 常见误解

### 误解 1：trainer 启动成功就说明前面没问题

不一定。

前面 preprocess 或数据契约的问题，经常会在训练表现上延迟暴露。

### 误解 2：`SegTrainer` 是一个大改版 trainer

不是。

它是项目默认 trainer 别名，主逻辑还是在 `nnUNetTrainer.py`。

### 误解 3：在线验证指标就是最终验证指标

不是。

在线验证更像训练期趋势观察，最终验证来自训练后正式导出结果。

### 误解 4：五折只是为了更“学术化”

不对。

五折在这里还承担：

- configuration selection
- ensemble 来源

---

## 11. 推荐实践顺序

对于当前项目，最稳妥的实践顺序是：

1. 先 `train --fold 0`
2. 看 `debug.json / training_log / progress.png`
3. 确认 validation 目录正常生成
4. 再 `train-all --npz`

这能帮你把：

- 工程链问题
  和
- 模型表现问题

尽量分开。

---

## 12. 训练阶段之后应该立刻接哪一步

训练阶段之后，最自然的下一站是：

- [04_inference_and_evaluation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/README.md)

因为后面真正要解决的是：

- 怎样把多折训练结果变成最终可用推理方案

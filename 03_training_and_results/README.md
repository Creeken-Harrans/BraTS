# 03 Training And Results

命令位置说明：
- 本文默认假设你当前目录就是 `BraTS` 项目根目录，因此命令示例写成 `python run.py ...`。
- 如果你当前在上一级目录 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径默认也都相对于 `BraTS` 项目根目录。

这一阶段负责把 preprocess 后的数据真正送入训练器，并产生日志、checkpoint、validation 结果和后续推理所需的模型产物。

这层的重点不是“命令怎么敲”，而是：

- trainer 是怎么被实例化的
- fold 在这里到底是什么意思
- 训练产物分别该怎么看
- 哪些文件只是过程观察用，哪些是后续推理必须依赖的

---

## 1. 当前项目的默认训练约定

默认组合是：

- `trainer = SegTrainer`
- `plans = ProjectPlans`
- `configuration = 3d_fullres`
- `num_epochs = 100`

这三个默认值不是随便挑的，而是和前一阶段的 planner 结果对应起来的本地约定。

对你当前项目而言，这意味着：

- 文档主线围绕 `SegTrainer + ProjectPlans + 3d_fullres`
- CLI 默认值也围绕这个组合
- 同一个 fold 如果已经有 checkpoint，默认会自动续训

---

## 2. 输入、输出和命令

### 输入

- preprocess 后的数据目录
- `ProjectPlans.json`
- `dataset.json`
- `splits_final.json`

### 运行命令

先验证链路：

```bash
python run.py train --fold 0
```

补充说明：

- 默认会训练 `100` 轮
- 如果 `fold_0` 已经有旧训练结果，会自动判断是续训还是新开
- 自动续训时的 checkpoint 选择顺序是：`checkpoint_final` -> `checkpoint_latest` -> `checkpoint_best`
- 如果你想强制从头开始，要显式加 `--restart-training`
- 无论是续训还是新开，都会把决策写进 `training_log_*.txt`
- 每个 epoch 的训练曲线会持续更新到 `progress.png`
- `checkpoint_latest` 现在会在每个 epoch 后覆盖更新，减少异常中断时的进度回退

再跑完整五折：

```bash
python run.py train-all --npz
```

`train-all` 也遵循同样规则：

- 某个 fold 有可用 checkpoint，就接着那个 fold 继续
- 某个 fold 没有可用 checkpoint，就从头开始训练该 fold
- 每个 fold 都各自写自己的 `training_log_*.txt` 和 `progress.png`

必要时只补验证：

```bash
python run.py train --fold 0 --validation-only --npz
```

### 输出

真正新训练的结果会写到：

- `PROJECT_RESULTS/Dataset220_BraTS2020/...`

而当前目录下保留的：

- `results/fold_0/`

主要是一个历史样例快照，帮助你理解产物结构。

---

## 3. 对应哪些代码

训练调度：

- [run_training.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/run/run_training.py)

默认 trainer 别名：

- [SegTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/SegTrainer.py)

trainer 主体：

- [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)

数据加载：

- [data_loader.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/dataloading/data_loader.py)
- [nnunet_dataset.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/dataloading/nnunet_dataset.py)

日志与损失：

- [nnunet_logger.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/logging/nnunet_logger.py)
- [compound_losses.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/loss/compound_losses.py)

如果你现在不是想看某个单文件，而是想按目录逐个理解整个代码包，直接看：

- [code_directory_audit.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/docs/code_directory_audit.md)

这份文档会把 `src/brats_project/` 拆成：

- 入口与调度层
- 主算法流水线层
- 扩展与兼容层
- 缓存与非源码层

它的作用是先帮你分清“当前默认主链”和“保留扩展目录”，避免一开始就陷进 trainer 变体和兼容脚本。

---

## 4. 训练调用链应该怎样理解

真实调用链大致是：

1. `run.py`
2. `cli.py`
3. `run_training.py`
4. `get_trainer_from_args(...)`
5. `SegTrainer(...)`
6. `nnUNetTrainer.initialize()`
7. `nnUNetTrainer.run_training()`

这条链意味着：

- 项目层负责组织命令、路径和默认值
- trainer 层负责真正训练

所以当你读训练日志和看代码时，最好把：

- “命令入口问题”
- “trainer 内部问题”

分开理解。

另外现在有一个很重要的实际行为：

- CLI 会在进入 trainer 之前先检查当前 configuration 的 preprocess 输出是否完整
- 如果训练结果目录里已经存在 checkpoint，则默认把这次调用视为“继续训练”
- 因此 `python run.py train --fold 0` 在当前项目里更接近“确保 fold_0 处于继续可跑状态”，而不是机械地每次都从 epoch 0 开始

---

## 5. trainer 初始化时依赖什么

trainer 初始化至少依赖下面这些输入：

- 数据集名或 ID
- configuration
- fold
- trainer 名字
- plans 名字

然后系统会去读：

- `PROJECT_PREPROCESSED/Dataset220_BraTS2020/ProjectPlans.json`
- `PROJECT_PREPROCESSED/Dataset220_BraTS2020/dataset.json`

并据此构造：

- `PlansManager`
- `ConfigurationManager`
- `LabelManager`
- 网络
- loss
- dataloader

也就是说，训练能否真正启动，至少要建立在这三个阶段都已正确完成的前提上：

1. 数据准备完成
2. preprocess 完成
3. 当前 trainer/plans/configuration 名称与真实文件一致

---

## 6. `fold` 在这里到底是什么意思

很多人第一次看训练命令时，会把 `fold` 误解成“第几次训练”。

实际上，`fold` 代表：

- 从 `splits_final.json` 中取出的某一组训练/验证划分

也就是说：

- `fold_0`
  - 是第一组 train/val 划分
- `fold_1`
  - 是第二组划分
- ...

所以先跑 `fold_0` 的真正意义不是“先试一次”，而是：

- 用第一组交叉验证划分先验证整条训练链

---

## 7. 为什么建议先跑 `fold_0`

这是当前项目最重要的工程实践建议之一。

### 因为你最先要排除的是链路错误

包括：

- trainer 类名不对
- plans 文件找不到
- preprocess 目录不对
- 输出目录不可写
- checkpoint 逻辑异常
- GPU 不可见

这些都不需要五折才能暴露。

### 因为五折训练成本很高

先用 `fold_0` 把工程侧问题排干净，比一开始就 `train-all` 更稳妥。

---

## 8. `--npz` 为什么重要

完整五折训练时，推荐：

```bash
python run.py train-all --npz
```

原因不是“多存点东西保险”，而是因为后面的：

- best configuration selection
- ensembling

都依赖 validation 概率输出。

如果没有 `.npz` 概率文件：

- 单模型 crossval 还能汇总
- 但 ensemble 会失去基础材料

---

## 9. trainer 主体里真正发生了什么

当前默认 trainer 其实就是：

- `SegTrainer -> nnUNetTrainer`

所以真正的训练核心都在 `nnUNetTrainer.py` 中。

它负责：

1. 初始化网络和优化器
2. 根据 plans 读取 configuration
3. 构造训练/验证 dataloader
4. 配置数据增强
5. 做 epoch 循环
6. 做在线验证
7. 写 checkpoint 和 log
8. 训练结束后做真实 validation 导出

你可以把它理解成：

- “训练总控类”

---

## 10. 你应该怎么读训练产物

### `debug.json`

这是初始化状态快照。

适合回答：

- 实际用了哪份 plans
- 当前输出目录是什么
- 当前设备是什么
- dataloader 和 transform 是什么

### `training_log_*.txt`

这是过程日志。

适合回答：

- 训练是否真正开始
- 当前 epoch 在哪
- checkpoint 什么时候写
- validation 什么时候触发

### `progress.png`

这是趋势图。

适合快速看：

- train loss
- val loss
- pseudo dice 变化

但它不能替代后面的：

- 真实 validation summary
- cross-validation 汇总结果

---

## 11. 当前目录中的 `results/fold_0/` 应该怎么用

这一目录不是为了直接复用旧模型，而是为了帮助你理解训练输出结构。

它的价值主要在于：

- 让你知道新训练跑完后去哪里找什么
- 让你先熟悉产物类别，而不是先追求旧结果好坏

建议你重点看：

- `debug.json`
- `training_log_*.txt`
- `progress.png`

---

## 12. 这一阶段和后面推理的关系

训练阶段不是终点。

后面推理和最佳配置选择至少依赖这里的这些东西：

- 训练好的 checkpoint
- `plans.json`
- `dataset.json`
- `validation` 结果
- 如果要 ensemble，还需要 `.npz` 概率输出

所以这一步的很多产物其实是在为下一阶段铺路。

---

## 13. 常见误解

### 误解 1：`SegTrainer` 是一个特别复杂的新 trainer

不是。

它只是项目默认 trainer 名字，真正逻辑主要在 `nnUNetTrainer.py`。

### 误解 2：`progress.png` 就代表模型最终质量

不对。

它只是过程趋势图，不能代替：

- 真实 validation
- 多折汇总
- best configuration selection

### 误解 3：训练目录里所有文件都同等重要

不对。

有些是过程观察用，有些是后续推理必须依赖的。

### 误解 4：只要训练能跑起来，就说明前面都没问题

不一定。

很多前面阶段的问题会以“训练表现异常”的形式延迟暴露。

---

## 14. 建议的阅读顺序

如果你现在准备继续深读训练相关代码，推荐顺序：

1. [run_training.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/run/run_training.py)
2. [nnUNetTrainer.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/nnUNetTrainer/nnUNetTrainer.py)
3. [data_loader.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/training/dataloading/data_loader.py)
4. [training_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/docs/training_guide.md)

这样你会更容易把：

- 命令入口
- trainer 生命周期
- dataloader 采样逻辑
- 训练产物

连成一条线。

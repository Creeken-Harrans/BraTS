# 02 Preprocess

这一阶段是整套项目里最接近 nnU-Net 核心思想的一层。

如果说：

- `01_data_preparation`
  - 负责把原始 BraTS 病例整理成合法训练目录

那么这一层负责的就是：

- 理解这份数据集
- 为这份数据集自动生成训练方案
- 把原始训练目录转换成训练器真正读取的预处理数据

一句话概括：

- 这一步是从“数据目录”走向“可训练表示”的桥。

---

## 1. 这一阶段完成什么

它实际做四件事：

1. 提取数据集 fingerprint
2. 根据 fingerprint 生成 plans
3. 根据 plans 执行 preprocess
4. 准备交叉验证 split 与 GT 备份

对应的典型产物有：

- `dataset_fingerprint.json`
- `ProjectPlans.json`
- `splits_final.json`
- 预处理后的 case 数据目录

---

## 2. 输入、输出和命令

### 输入

- `PROJECT_RAW/Dataset220_BraTS2020`
- `dataset.json`

### 输出

默认写到：

- `PROJECT_PREPROCESSED/Dataset220_BraTS2020`

### 运行命令

```bash
python BraTS/run.py plan-preprocess
```

常用参数示例：

```bash
python BraTS/run.py plan-preprocess \
  --verify-dataset \
  --plans ProjectPlans \
  --configurations 3d_fullres
```

默认行为：

- 已有 `dataset_fingerprint.json` 时直接复用
- 已有 `ProjectPlans.json` 时直接复用
- 已有某个 configuration 的预处理输出时直接复用
- 每个 case 的详细 preprocess 输出写入 `02_preprocess/logs/preprocess_*.log`
- `02_preprocess/logs/latest.log` 始终指向最近一次 preprocess 日志

只有在你明确要求时才重算：

```bash
python BraTS/run.py plan-preprocess --recompute-fingerprint
python BraTS/run.py plan-preprocess --recompute-plans
python BraTS/run.py plan-preprocess --force-preprocess
python BraTS/run.py plan-preprocess --clean
```

其中：

- `--recompute-fingerprint` 只重算 fingerprint
- `--recompute-plans` 只重算 plans
- `--force-preprocess` 只重做预处理输出
- `--clean` 三者都重做

---

## 3. 对应哪些代码

CLI 调度：

- [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)

API 入口：

- [plan_and_preprocess_api.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/plan_and_preprocess_api.py)

fingerprint：

- [fingerprint_extractor.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/dataset_fingerprint/fingerprint_extractor.py)

planner：

- [default_experiment_planner.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/experiment_planning/experiment_planners/default_experiment_planner.py)

preprocessor：

- [default_preprocessor.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/preprocessing/preprocessors/default_preprocessor.py)

plans 访问层：

- [plans_handler.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/utilities/plans_handling/plans_handler.py)

---

## 4. 这一阶段按真实顺序在做什么

### 第一步：验证数据完整性

如果使用 `--verify-dataset`，系统会先检查：

- 通道数是否符合 `dataset.json`
- 标签值是否合法
- geometry 是否一致

这一步是正式 preprocess 前的输入验收。

### 第二步：提取 fingerprint

fingerprint 会对整个数据集做“数据画像”，主要统计：

- spacing 分布
- 裁剪后 shape
- 裁剪后体积比例
- 前景强度统计

它回答的是：

- 这份数据整体长什么样

而不是：

- 某一例长什么样

### 第三步：planner 根据 fingerprint 生成 plans

planner 会根据 fingerprint 决定：

- target spacing
- transpose 方向
- patch size
- batch size
- 哪些 configuration 存在
- 使用哪种 normalization/resampling/preprocessor

你可以把 planner 理解成：

- “把数据画像翻译成训练配方的模块”

### 第四步：preprocessor 按 plans 真正处理病例

对每个 case，preprocessor 大致做：

1. 读图像和标签
2. 转置轴顺序
3. 裁剪到非零区域
4. 计算目标 shape
5. 做归一化
6. 做重采样
7. 记录前景采样位置
8. 保存成训练期快速读取格式

### 第五步：准备 split 和 GT

系统还会：

- 生成或复制 `splits_final.json`
- 复制 `gt_segmentations/`

这样后面即使 raw 数据不在，也还能做 validation 与评估。

---

## 5. 这一阶段最关键的三个文件

### `dataset_fingerprint.json`

这是 planner 的输入依据。

看它时最值得关注：

- `spacings`
- `shapes_after_crop`
- `median_relative_size_after_cropping`
- `foreground_intensity_properties_per_channel`

### `ProjectPlans.json`

这是 planner 给出的正式决议。

看它时最值得关注：

- `configurations`
- `spacing`
- `patch_size`
- `batch_size`
- `normalization_schemes`
- `resampling_fn_*`
- `architecture`

### `splits_final.json`

这是后面训练时每个 fold 的 train/val 划分来源。

如果你不理解它，后面 `fold_0` 到 `fold_4` 在你脑子里就只会是编号，而不是实际的数据划分。

---

## 6. 当前 BraTS 项目在这一层的核心意义

当前项目的主线不是“手工调一堆超参数”，而是先让 planner 对当前 BraTS 数据做自动配置。

这意味着：

- `ProjectPlans.json` 不是随便抄来的模板
- 它是当前数据集、当前 planner 和当前显存目标共同作用的结果

所以你最该问的问题不是：

- “为什么 batch size 只有 2？”

而是：

- “当前 fingerprint 和 planner 是怎样推导出这个 batch size 的？”

---

## 7. 为什么不要一上来就改 plans

这是这一阶段最常见的误区之一。

### 错误心态

- 看到 `ProjectPlans.json` 就直接手工改 `patch_size`
- 看到 `batch_size=2` 就直接想改大

### 为什么这样危险

因为你还没先理解：

- 当前 spacing 为什么这样选
- patch 为何需要是这个大小
- 显存预算和网络拓扑怎样约束 batch size
- preprocess 结果是否已经和这套 plans 一致

也就是说，plans 不是“手调参数速记本”，而是“planner 的正式决议文件”。

如果要改，应该先知道你在推翻哪一层依据。

---

## 8. 为什么保留 `metadata/` 样例很有价值

这一阶段的产物很多，而且它们本身又是后续阶段输入。

项目把一份样例快照保留在：

- `02_preprocess/metadata/`

它的价值是：

- 你不用每次都重跑才能知道当前数据画像长什么样
- 你可以在读代码时随时对照真实结果文件
- 你可以在不动训练环境的情况下先理解 planner 产物

这些保留下来的样例，本质上相当于：

- “对一次成功 preprocessing 的结构化快照”

---

## 9. 这一阶段和后续训练的关系

trainer 并不直接吃 raw NIfTI。

它真正吃的是：

- preprocess 后的数据文件
- `ProjectPlans.json`
- `dataset.json`
- `splits_final.json`

所以很多训练阶段看起来像 trainer 的问题，根因其实在这一层：

- patch 尺度不合理
- class_locations 缺失
- geometry 还原链断裂
- normalization 策略不合适

---

## 10. 常见误解

### 误解 1：preprocess 就是 resize

不是。

它是：

- 裁剪
- 归一化
- 重采样
- 目标构造
- 前景位置记录
- 格式转换

的组合过程。

### 误解 2：fingerprint 是可有可无的统计文件

不是。

它是 planner 的输入依据。

### 误解 3：plans 只影响训练阶段

不是。

plans 还会影响：

- preprocess
- inference
- prediction export

### 误解 4：split 只是保存方便

不是。

它直接定义了后面 trainer 的 fold 语义。

---

## 11. 建议检查顺序

如果你刚跑完这一阶段，建议按下面顺序检查：

1. `metadata/dataset.json`
2. `metadata/dataset_fingerprint.json`
3. `metadata/ProjectPlans.json`
4. `metadata/splits_final.json`

这个顺序的好处是：

- 先确认任务定义
- 再确认数据画像
- 再确认训练方案
- 最后确认数据划分

这比一上来就只盯着 `ProjectPlans.json` 更不容易误判。

---

## 12. 下一步该看哪里

如果你已经理解了这一层，下一步建议去看：

- [02_preprocess/docs/preprocess_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/docs/preprocess_guide.md)
- [03_training_and_results/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/README.md)

因为接下来真正要进入的是：

- 训练器如何消费这套 preprocess 产物

# 01 Data Preparation

命令位置说明：
- 本文默认假设你当前目录就是 `BraTS` 项目根目录，因此命令示例写成 `python run.py ...`。
- 如果你当前在上一级目录 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径默认也都相对于 `BraTS` 项目根目录。

这一阶段做的不是“复制文件”，而是把原始 BraTS 病例转换成当前项目训练链路真正承认的输入目录。

如果把整个项目看成一条流水线，那么这一层就是：

- 从原始 BraTS 表示
  ->
- 到本项目训练表示

的桥。

---

## 1. 这一阶段的目标

它要解决四个问题：

1. 原始病例目录如何转换成 `Dataset220_BraTS2020`
2. 四模态怎样稳定映射到固定通道顺序
3. 原始标签 `0/1/2/4` 怎样转换成项目训练标签
4. `dataset.json` 怎样定义 region-based training 所需的任务语义

换句话说，这一层不是“文件整理”，而是“数据契约建立”。

---

## 2. 输入、输出和入口

### 输入

- 原始 BraTS 根目录
- 每个病例包含：
  - `t1`
  - `t1ce`
  - `t2`
  - `flair`
  - `seg`

### 输出

默认输出到：

- `PROJECT_RAW/Dataset220_BraTS2020`

其结构至少包括：

- `imagesTr/`
- `labelsTr/`
- `dataset.json`

### 运行命令

```bash
python run.py prepare-dataset
```

也可以显式指定：

```bash
python run.py prepare-dataset \
  --src-root ../BraTS-Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --project-raw ../BraTS-Dataset/nnUNet_raw
```

当前实现还有一个重要约定：

- 如果 `PROJECT_RAW/Dataset220_BraTS2020` 已经完整存在，命令会直接复用已有数据集并成功退出
- 只有在你显式传入 `--force` 时，才会先删除旧目录再重建

也就是说，重复执行时直接运行默认命令即可：

```bash
python run.py prepare-dataset
```

只有确实要重建时，才运行：

```bash
python run.py prepare-dataset --force
```

---

## 3. 对应哪些代码

项目入口：

- [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)

CLI 调度：

- [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)

核心脚本：

- [prepare_brats2020_for_project.py](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/scripts/prepare_brats2020_for_project.py)

底层数据契约说明：

- [data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/docs/data_contract.md)

保留的原始 dataset 样例：

- [raw_dataset.json](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/metadata/raw_dataset.json)

---

## 4. 代码按执行顺序在做什么

### 第一步：找到原始数据根和目标 raw 根

脚本会先解析：

- 原始 BraTS 根目录
- 目标 raw 数据集根目录

这里会优先尊重：

- CLI 参数
- `PROJECT_RAW`
- 默认配置

如果目标目录已经存在，脚本会先判断它是否已经是一个完整可复用的 `Dataset220_BraTS2020`：

- 完整则直接复用
- 不完整或不一致则报错
- 显式传 `--force` 时才会删除后重建

### 第二步：遍历并识别合法病例

脚本会扫描原始根目录下的子目录，判断哪些目录真正符合 BraTS case 结构。

它不是简单地看文件数，而是会明确寻找：

- `t1`
- `t1ce`
- `t2`
- `flair`
- `seg`

### 第三步：验证 geometry

对每个候选病例，脚本会检查：

- size
- spacing
- origin
- direction

为什么要在这里严格校验：

- 后面 preprocess 会把多模态直接按通道堆叠
- trainer 默认标签和图像逐 voxel 对齐
- 如果这里 geometry 不一致，后面的问题会更隐蔽

### 第四步：写出 4 个模态到 `imagesTr`

脚本会按照固定顺序输出：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

这一步最重要的不是文件名好看，而是：

- 后面训练和推理完全依赖这个顺序契约

### 第五步：把原始 BraTS 标签映射成项目训练标签

原始 BraTS 标签：

- `0 = background`
- `1 = NCR/NET`
- `2 = ED`
- `4 = ET`

项目训练标签：

- `0 -> 0`
- `2 -> 1`
- `1 -> 2`
- `4 -> 3`

这一映射不是随便为了“连续整数”而做，而是为了和后面的 region-based training 定义对齐。

### 第六步：生成 `dataset.json`

生成的 `dataset.json` 不只描述文件结构，还定义了任务语义：

- channel names
- labels
- `regions_class_order`
- `file_ending`
- `numTraining`

对当前项目来说，这个文件是后面：

- planner
- preprocessor
- label manager
- trainer

共同依赖的任务定义入口。

---

## 5. 当前数据集的核心契约

### 通道契约

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

这四个通道顺序必须全局固定。

### 标签契约

转换后训练标签不再是原始 BraTS 数字本身，而是项目内部重新编码后的标签：

- `0`
- `1`
- `2`
- `3`

### region 契约

当前项目训练目标不是直接把 4 个整数标签当作终点，而是组织成：

- `whole_tumor = [1, 2, 3]`
- `tumor_core = [2, 3]`
- `enhancing_tumor = [3]`

并且：

- `regions_class_order = [1, 2, 3]`

这个顺序不能乱。

### geometry 契约

同一病例内：

- 4 个模态必须对齐
- 标签必须与图像对齐

这不是推荐项，而是硬前提。

---

## 6. 这一阶段产物的下游用途

### 给 `02_preprocess`

它提供：

- 合法的 `Dataset220_BraTS2020`
- 正确的 `dataset.json`

没有这一步，fingerprint 和 planner 都没有稳定输入。

### 给 `03_training_and_results`

它间接决定：

- trainer 最终看到多少输入通道
- label manager 如何理解 region

### 给 `04_inference_and_evaluation`

它决定正式推理输入的命名标准应该是什么。

---

## 7. 最值得检查的文件

建议检查顺序：

1. `PROJECT_RAW/Dataset220_BraTS2020/imagesTr`
2. `PROJECT_RAW/Dataset220_BraTS2020/labelsTr`
3. `PROJECT_RAW/Dataset220_BraTS2020/dataset.json`
4. [raw_dataset.json](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/metadata/raw_dataset.json)

这样看最容易确认：

- 通道编号是否稳定
- 标签是否都被写出
- `dataset.json` 的 region 语义是否符合预期

---

## 8. 常见误解

### 误解 1：这一步只是重命名

不是。

它还负责：

- geometry 校验
- 标签重编码
- region 任务定义

### 误解 2：只要能生成 `imagesTr/labelsTr` 就算完成

不够。

真正关键的是：

- 通道顺序对不对
- 标签映射对不对
- `dataset.json` 语义对不对

### 误解 3：`dataset.json` 只是说明文件

不是。

它是后面代码真正会读取的任务定义文件。

### 误解 4：标签映射只是为了“连续数字更好看”

不只是。

它和 BraTS 的 region-based training 表达方式强相关。

---

## 9. 下一步该看哪里

如果你已经理解了这一层，下一步应该直接去看：

- [data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/docs/data_contract.md)
- [02_preprocess/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/README.md)

因为接下来项目就会从：

- “数据目录是否合法”

进入：

- “系统怎样自动理解这份数据并给出训练方案”

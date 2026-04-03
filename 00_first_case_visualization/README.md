# 00 First Case Visualization

命令位置说明：
- 本文默认假设你当前目录就是 `BraTS` 项目根目录，因此命令示例写成 `python run.py ...`。
- 如果你当前在上一级目录 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径默认也都相对于 `BraTS` 项目根目录。

这一阶段不是训练必须步骤，但它是整套项目里最值得先做的“认知准备”。

它解决的不是模型问题，而是数据理解问题：

- 原始 BraTS 病例到底长什么样
- 四个模态和标签是否真的共空间对齐
- 不同模态分别更容易看见哪些病灶信息
- 后面训练和推理默认成立的数据前提，你肉眼是否已经确认过

如果这一步没有建立直觉，后面你很容易把：

- preprocess
- plans
- trainer
- inference

都看成黑箱。

---

## 1. 这一阶段的输入、输出和入口

### 输入

- 原始 BraTS 数据根目录
- 每个病例应该至少包含：
  - `t1`
  - `t1ce`
  - `t2`
  - `flair`
  - `seg`

### 输出

默认输出目录：

- `BraTS/00_first_case_visualization/output`

典型产物：

- `case_summary.txt`
- `case_summary.json`
- `seg_labels_summary.txt`
- `intensity_stats.csv`
- 多张切片、overlay、bbox、直方图 PNG

### 运行命令

```bash
python run.py visualize-first-case
```

常见参数：

```bash
python run.py visualize-first-case \
  --data-root archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --output-dir BraTS/00_first_case_visualization/output
```

---

## 2. 对应哪些代码

主入口：

- [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)

CLI 调度：

- [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)

真正执行脚本：

- [visualize_first_case.py](/home/Creeken/Desktop/machine-learning-test/BraTS/00_first_case_visualization/visualize_first_case.py)

你可以把这三层理解成：

- `run.py`
  - 项目入口壳
- `cli.py`
  - 子命令分发器
- `visualize_first_case.py`
  - 真正干活的可视化与检查脚本

---

## 3. 脚本按执行顺序到底在做什么

### 第一步：解析工作区路径

脚本会先找到 `machine-learning-test` 工作区根目录，再把相对路径解释成相对于工作区的绝对路径。

这一步的意义不是“写法统一”这么简单，而是确保：

- 可视化阶段和后面的数据转换、预处理、训练使用同一套路径语义

### 第二步：递归扫描原始数据目录

脚本不是写死某个病例 ID，而是：

- 递归扫描原始数据根目录
- 识别 BraTS 风格 NIfTI 文件
- 自动找出第一个合法病例目录

这意味着它本质上更像：

- 一个数据探索器
- 一个数据结构检查器

而不是“一次性演示脚本”。

### 第三步：识别文件角色

脚本会自动判断每个 NIfTI 文件是：

- `t1`
- `t1ce`
- `t2`
- `flair`
- `seg`

这一步会抓出两类常见问题：

- 病例缺模态
- 同一病例下有多个文件同时匹配同一角色

### 第四步：加载 volume 并验证对齐

脚本会读取 3D volume，并检查：

- shape
- spacing
- affine

为什么这是硬检查：

- overlay 要可信，模态和标签必须在同一空间
- 后面的 `prepare-dataset` 和训练也默认这个前提成立

### 第五步：统计标签和强度

脚本会生成：

- segmentation label 统计
- 各模态强度统计

这一步的价值在于：

- 它先让你看到数据的数值分布和标签占比，而不是等到训练 loss 很怪时再猜原因

### 第六步：生成多种图像

脚本会生成多类图：

- 中间切片三视图
- 各模态 montage
- segmentation montage
- flair / t1ce overlay
- tumor bbox 全局与局部视图
- 强度直方图
- 标签分布图

这些图不是重复劳动，而是在从不同视角解释同一个 3D 病例：

- 空间位置
- 模态差异
- 标签分布
- 病灶局部形态

---

## 4. 你应该怎么读输出目录

推荐顺序：

1. `output/case_summary.txt`
2. `output/seg_labels_summary.txt`
3. `output/modalities_mid_slices.png`
4. `output/t1_montage.png`
5. `output/t1ce_montage.png`
6. `output/t2_montage.png`
7. `output/flair_montage.png`
8. `output/seg_montage.png`
9. `output/flair_overlay_best_slices.png`
10. `output/t1ce_overlay_best_slices.png`
11. `output/tumor_bbox_views.png`
12. `output/intensity_histograms.png`
13. `output/seg_label_distribution.png`

为什么这样读最有效：

- 先用文本搞清楚文件和标签
- 再建立三维空间和模态直觉
- 再把标签叠回影像看
- 最后看数值分布和类别分布

---

## 5. 这一阶段最值得掌握的几个概念

### 3D volume

BraTS 不是普通图片数据集，而是三维体数据。

你后面在 trainer、predictor 里看到的所有：

- patch
- spacing
- resampling
- sliding window

都建立在这个前提上。

### 多模态共空间

四个模态不是四个独立样本，而是同一个病例在不同 MRI 模态下的共空间表示。

所以模型输入不是“4 张不同图片”，而是：

- 一份 4 通道 3D 体数据

### segmentation 也是 3D volume

`seg` 不是轮廓图，也不是单张 mask。

它是逐 voxel 的三维标签 volume，后面训练监督和推理导出都在依赖这一点。

---

## 6. 这一阶段和后面各阶段的关系

### 和 `01_data_preparation` 的关系

这一阶段先帮你确认：

- 原始模态角色是否清楚
- geometry 是否合理
- label 语义是否清楚

否则你很难判断数据转换后是不是变对了。

### 和 `02_preprocess` 的关系

这一阶段让你知道：

- 原始 spacing 和 shape 是什么
- 不同模态强度分布怎样

这样你后面再看 `dataset_fingerprint.json` 和 `ProjectPlans.json` 时，才知道 planner 为什么做出那些选择。

### 和 `03_training_and_results` 的关系

这一阶段让你知道 trainer 看到的 patch 在原始数据里大概来自什么样的病灶空间结构。

### 和 `04_inference_and_evaluation` 的关系

这一阶段让你在看预测结果时，不会把一个三维结构当成二维图片去理解。

---

## 7. 常见误解

### 误解 1：这只是“看图玩”的脚本

不是。

它实际上是整个项目的数据 sanity-check 第一层。

### 误解 2：只看中间切片就够了

不够。

病灶最明显的位置未必在中间层，所以脚本才会额外生成：

- montage
- best slices overlay
- bbox views

### 误解 3：看起来能叠上去就说明数据没问题

不一定。

你还需要看：

- spacing
- affine
- 标签值
- 各模态强度分布

### 误解 4：这是训练前的可选装饰

它确实不是训练命令必需步骤，但对“真正理解项目”来说几乎是必需步骤。

---

## 8. 最后建议

如果你现在要继续往下读项目，最好的下一站是：

- [01_data_preparation/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/README.md)

因为这一步正好承接：

- “我已经知道原始病例长什么样”
  ->
- “我现在要把它变成训练目录”

# Terms and Theory

这份文档专门负责三件事：

1. 把项目里反复出现的英文术语译成稳定的中文语义。
2. 解释这些词在 BraTS 和 nnU-Net 语境里到底指什么。
3. 让你在读源码和日志时，不至于因为概念混乱把工程问题和算法问题搅在一起。

这不是论文复述，也不是源码逐行解释，而是一份“概念地图”。

---

## 1. 整个项目的核心理论框架

### 自配置方法

- 中文：自配置方法
- 英文：self-configuring method

这是 nnU-Net 的总思想。

它不是先拍脑袋固定一套网络、patch size 和 batch size，再让数据集去适应；它是先分析数据集，再为这个数据集生成一套相对合适的训练方案。

在当前项目里，这个思想落到三个实体上：

1. `dataset_fingerprint.json`
   - 数据画像
2. `ProjectPlans.json`
   - 训练与预处理方案
3. `trainer / preprocessor / predictor`
   - 按 plans 执行的具体组件

所以当你看到：

- fingerprint
- plans
- 3d_fullres
- patch size
- spacing

这些词时，脑子里应该把它们串成：

```text
先理解数据 -> 再决定方案 -> 再执行训练与推理
```

### 流水线

- 中文：流水线、处理链路、管线
- 英文：pipeline

在这个项目里，pipeline 指从原始 MRI 到最终预测和评估的整条顺序化过程，而不是单个脚本。

当前项目主线是：

1. 原始病例理解
2. 数据转换
3. fingerprint
4. experiment planning
5. preprocess
6. training
7. validation
8. best configuration selection
9. inference
10. postprocessing
11. evaluation

### 工程外壳与算法内核

这是理解这个项目时非常重要的一层概念区分。

工程外壳负责：

- 项目路径
- 命令入口
- 本地默认配置
- 数据转换和文档组织

算法内核负责：

- experiment planning
- preprocessing
- dataloading
- training
- inference
- evaluation

所以不要把：

- `run.py / cli.py / project_config.json`

和：

- `nnUNetTrainer.py / default_preprocessor.py / predict_from_raw_data.py`

混成一层。

---

## 2. BraTS 数据相关术语

### 病例

- 中文：病例、个案
- 英文：case

一个 case 指一个病人的完整输入集合，包括：

- 4 个 MRI 模态
- 1 个 segmentation 标签文件

在工程层面，case 是最小组织单位：

- split 按 case 划分
- 训练输出按 case 验证
- 推理输入按 case 组织

### 病例标识符

- 中文：病例标识符
- 英文：case identifier

例如：

- `BraTS20_Training_001`

它用于把：

- `BraTS20_Training_001_0000.nii.gz`
- `BraTS20_Training_001_0001.nii.gz`
- `BraTS20_Training_001_0002.nii.gz`
- `BraTS20_Training_001_0003.nii.gz`
- `BraTS20_Training_001.nii.gz`

这些文件关联为同一个病例。

### 模态

- 中文：模态
- 英文：modality

模态是医学成像意义上的不同成像方式。

BraTS 当前项目使用四模态：

- `T1`
- `T1ce`
- `T2`
- `FLAIR`

它们不是四张不同角度的图片，而是四个已经对齐到同一空间的 3D volume。

### 通道

- 中文：通道
- 英文：channel

通道是模型输入层面的编号表达。

在本项目的数据契约里：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

所以：

- modality 是医学语义
- channel 是模型输入顺序

### 分割

- 中文：分割
- 英文：segmentation

分割就是给图像中的每个 voxel 指定语义类别或区域归属。

在 BraTS 场景下，分割不只是“肿瘤 vs 背景”，而是带有内部子结构区分的病灶分割。

### 标签

- 中文：标签
- 英文：label

label 既可以指：

- 一个整数值
- 这个整数值对应的语义

原始 BraTS 标签通常是：

- `0 = background`
- `1 = NCR/NET`
- `2 = ED`
- `4 = ET`

而本项目训练目录内部会先做一次标签重编码。

### 区域

- 中文：区域
- 英文：region

region 和 label 不同。

label 通常是互斥类别；region 则可以是多个 label 的组合。

本项目里：

- `whole_tumor = [1, 2, 3]`
- `tumor_core = [2, 3]`
- `enhancing_tumor = [3]`

region 是 BraTS 任务里最关键的概念之一，因为官方评估关注的就是这些区域。

### 体素

- 中文：体素
- 英文：voxel

voxel 是三维图像中的最小单位，可以理解成三维版 pixel。

但 voxel 比 pixel 多一层非常关键的含义：

- 它对应真实物理空间中的一个小体积单元

所以模型处理的不是普通图片堆，而是带空间意义的 3D volume。

### 体素间距

- 中文：体素间距
- 英文：voxel spacing

spacing 表示相邻 voxel 在真实空间中的距离，通常单位是毫米。

它之所以关键，是因为：

- 不同病例 spacing 可以不同
- 同样的 `128 x 128 x 128` patch，在不同 spacing 下对应的真实物理范围不一样
- preprocess 里的 resampling 就是在统一这个尺度

### 仿射矩阵

- 中文：仿射矩阵
- 英文：affine

affine 用来描述：

- 数组索引坐标怎样映射到真实世界坐标

在 3D 医学影像中，它是 geometry 的一部分。

### 几何一致性

- 中文：几何一致性
- 英文：geometry consistency

同一病例内，多模态和标签必须在几何上匹配。

通常至少要求：

- shape 一致
- spacing 一致
- origin 一致
- direction 或 affine 一致

为什么这是一条硬约束：

- 多模态需要按 channel 直接堆叠
- 标签监督默认逐 voxel 对齐
- overlay 和逆导出都依赖这个前提

### 配准

- 中文：配准
- 英文：registration / co-registration

多模态图像被对齐到同一空间的过程叫配准。

在这个项目里，配准不是训练时再去学的东西，而是输入前提。

也就是说：

- 模型默认你喂进去的数据已经完成 co-registration

---

## 3. BraTS 任务相关术语

### 全肿瘤

- 中文：全肿瘤
- 英文：Whole Tumor, WT

表示病灶总体范围，通常包含：

- edema
- tumor core
- enhancing tumor

在本项目的训练定义里，对应：

- `whole_tumor = [1, 2, 3]`

### 肿瘤核心

- 中文：肿瘤核心
- 英文：Tumor Core, TC

表示病灶内部更核心的部分，排除较外围水肿。

在本项目中，对应：

- `tumor_core = [2, 3]`

### 增强肿瘤

- 中文：增强肿瘤
- 英文：Enhancing Tumor, ET

表示在增强 MRI 上明显增强的区域。

在本项目中，对应：

- `enhancing_tumor = [3]`

### 水肿

- 中文：水肿
- 英文：edema, ED

原始 BraTS 标签之一，常在 FLAIR 上更明显。

### 坏死 / 非增强核心

- 中文：坏死 / 非增强肿瘤核心
- 英文：necrotic and non-enhancing tumor core, NCR/NET

原始 BraTS 标签之一，对应病灶核心内部的另一部分语义。

---

## 4. fingerprint 和 plans 相关术语

### 数据集指纹

- 中文：数据集指纹
- 英文：dataset fingerprint

它是对整个数据集特征的摘要，而不是对单个病例的摘要。

常见内容包括：

- spacing 分布
- shape 分布
- 裁剪后体积
- 前景强度统计

在本项目里对应：

- `dataset_fingerprint.json`

### 实验规划

- 中文：实验规划
- 英文：experiment planning

experiment planning 是从 fingerprint 生成 plans 的过程。

它回答的问题是：

- target spacing 该取多少
- patch size 该多大
- batch size 该多少
- 是否需要低分辨率配置
- 网络该采用怎样的层级结构

### plans 文件

- 中文：plans 文件、计划文件
- 英文：plans file

plans 文件是 planner 最终给出的正式配置决议。

在本项目里最常见的是：

- `ProjectPlans.json`

它会定义：

- 数据配置
- patch size
- batch size
- network architecture 参数
- normalization
- resampling
- preprocessor

你可以把它理解成：

- “当前数据集和当前训练方案之间的正式合同”

### 目标间距

- 中文：目标间距
- 英文：target spacing

表示 preprocess 后希望病例统一到怎样的 spacing。

它决定：

- 真实物理尺度是否统一
- resampling 后 shape 如何变化
- patch 在物理世界中覆盖多大范围

### patch size

- 中文：图像块大小
- 英文：patch size

patch size 是送入网络的局部 2D/3D 子块大小。

对 3D 分割来说，它直接决定：

- 网络每次看到多少上下文
- 显存占用有多高
- batch size 能否增大

### batch size

- 中文：批大小
- 英文：batch size

表示一次参数更新包含多少个 patch。

在 3D 医学图像里，它经常很小，因为显存非常紧张。

### transpose

- 中文：轴转置
- 英文：transpose

planner 和 preprocessor 中的 transpose 指：

- 为了统一 spacing 解释和网络处理方式，对数据轴顺序做变换

它不是视觉上的旋转，而是数组维度意义上的重排。

### 各向异性

- 中文：各向异性
- 英文：anisotropy

表示不同轴上的 spacing 差异明显。

它会影响：

- target spacing 选择
- dummy 2D augmentation
- resampling 策略

### 重采样

- 中文：重采样
- 英文：resampling

指把数据从原 spacing 变到 target spacing。

注意：

- 图像和分割的重采样策略不同
- 图像通常允许更高阶插值
- 分割必须更谨慎，避免标签边界被插值污染

### 归一化

- 中文：归一化
- 英文：normalization

归一化是把不同病例、不同模态的强度统计拉到更稳定的尺度范围。

对 BraTS 脑 MRI，这通常不仅仅是“全图标准化”，而是更强调：

- 非零区域
- 脑区前景
- mask-aware normalization

### `use_mask_for_norm`

- 中文：使用 mask 限制归一化区域
- 英文：use mask for normalization

表示归一化时是否只在前景/非零区域上计算统计量，并让外部区域保持为 0。

这在 BraTS 脑 MRI 任务中很常见，因为背景零值非常多。

---

## 5. 训练相关术语

### 训练器

- 中文：训练器
- 英文：trainer

trainer 不是单纯的网络，而是完整训练行为的组织者。

它通常负责：

- 初始化网络
- 初始化优化器和 lr scheduler
- 构造 dataloader
- 构造 loss
- epoch 循环
- validation
- checkpoint

当前项目默认 trainer 是：

- `SegTrainer`

但它本质上只是：

- `nnUNetTrainer` 的项目默认别名

### fold

- 中文：折
- 英文：fold

fold 指交叉验证中的某一组 train/val 划分。

当前项目默认做五折：

- `fold_0` 到 `fold_4`

### 交叉验证

- 中文：交叉验证
- 英文：cross-validation

交叉验证在这个项目里有两个角色：

1. 估计模型泛化表现
2. 给后续 configuration selection 和 ensembling 提供依据

所以它不是只为了一张好看的验证表格。

### 深度监督

- 中文：深度监督
- 英文：deep supervision

表示在 decoder 多个尺度输出上同时施加监督。

作用通常包括：

- 稳定优化
- 让中间尺度也学到语义

### 伪 Dice

- 中文：伪 Dice、在线 Dice
- 英文：pseudo Dice

这是训练过程中用当前 batch 或验证 batch 汇总出来的即时 Dice 指标。

它和最终 validation 文件夹里重新推理出的真实 Dice 不是一回事。

### checkpoint

- 中文：检查点
- 英文：checkpoint

保存训练状态的文件，通常包括：

- 网络权重
- 优化器状态
- grad scaler 状态
- 当前 epoch
- logger 状态

常见文件：

- `checkpoint_latest.pth`
- `checkpoint_best.pth`
- `checkpoint_final.pth`

### DDP

- 中文：分布式数据并行
- 英文：Distributed Data Parallel

表示多 GPU 并行训练方式。

当前项目里：

- 单卡是默认常态
- `num_gpus > 1` 时会走 DDP 分支

### 前景过采样

- 中文：前景过采样
- 英文：foreground oversampling

因为医学图像里背景通常远多于病灶，如果 patch 全随机采样，很多 batch 可能几乎全是背景。

所以 dataloader 会利用 `class_locations`：

- 有意识地抽更多包含前景的 patch

### 数据增强

- 中文：数据增强
- 英文：data augmentation

当前项目默认增强包括：

- 旋转
- 缩放
- mirror
- 噪声
- 模糊
- 对比度/亮度/gamma 扰动
- 低分辨率模拟

其目的是提高鲁棒性，而不是生成“更漂亮”的图像。

---

## 6. BraTS 特有训练语义

### label-based training

- 中文：标签式训练
- 英文：label-based training

把任务看成互斥类别预测，例如：

- background
- edema
- necrotic core
- enhancing tumor

这更接近传统多类分割。

### region-based training

- 中文：区域式训练
- 英文：region-based training

把训练目标直接定义为 BraTS 评估关注的区域，例如：

- WT
- TC
- ET

其特点是：

- 各 region 可以重叠
- 不再适合简单互斥 softmax 解释

当前项目的数据定义就是 region-based training 风格。

### `regions_class_order`

- 中文：区域类别顺序
- 英文：regions class order

这个字段非常关键。

它决定了 region 输出通道在还原 segmentation 时按什么顺序写回。

为什么它危险：

- 顺序如果被改乱，代码不一定立刻报错
- 但最终预测语义会错位

### sigmoid 与 softmax

- 中文：Sigmoid / Softmax
- 英文：sigmoid / softmax

直觉上：

- softmax 适合互斥类别
- sigmoid 适合彼此可重叠的目标

所以在 region-based training 场景下，sigmoid 更自然。

### BCE

- 中文：二元交叉熵
- 英文：binary cross-entropy, BCE

在 region-based training 场景下，常用 BCE 去独立优化每个 region head。

---

## 7. 推理、后处理与评估术语

### inference

- 中文：推理
- 英文：inference

指把训练好的模型应用到新病例上，生成 segmentation。

### validation

- 中文：验证
- 英文：validation

在本项目里通常有两层含义：

- 训练过程中在线验证
- 训练完成后对 held-out fold 做真实滑窗推理并导出结果

### sliding window prediction

- 中文：滑窗预测
- 英文：sliding window prediction

因为整幅 3D volume 往往放不进显存，所以推理时会把图像切成多个重叠 tile，逐块预测后再融合。

### Gaussian weighting

- 中文：高斯加权融合
- 英文：Gaussian weighting

滑窗预测时，为了减少 tile 边界伪影，会对 tile 中心区域给更高权重。

### 概率图

- 中文：概率图、概率输出
- 英文：probability maps

指模型输出的连续值结果，还没完全变成最终离散 segmentation。

这在后面做：

- fold 平均
- ensemble

时特别重要。

### ensemble

- 中文：集成
- 英文：ensemble

把多个模型的预测组合起来。

本项目主要是：

- 多 fold 概率平均
- 多 configuration 结果平均

### postprocessing

- 中文：后处理
- 英文：postprocessing

在模型输出之后再做规则型修正。

当前项目默认搜索的是：

- 是否移除小连通域
- 是否只保留最大连通域

### Dice

- 中文：Dice 系数
- 英文：Dice coefficient

衡量预测和 GT 的重叠程度。

它是 BraTS 和医学图像分割里最核心的指标之一。

### IoU

- 中文：交并比
- 英文：Intersection over Union

也是重叠指标，但在本项目里通常比 Dice 次一级常用。

### HD95

- 中文：95% 豪斯多夫距离
- 英文：95th percentile Hausdorff Distance

BraTS 官方评估里常出现的边界距离指标。

直觉上：

- Dice 看区域重叠
- HD95 看边界误差

### TP / FP / FN / TN

- 中文：真阳性 / 假阳性 / 假阴性 / 真阴性
- 英文：true positive / false positive / false negative / true negative

这些是 Dice、IoU 等指标的更底层组成部分。

---

## 8. 这些概念在项目目录里的落点

如果你想把“概念”和“目录”直接绑起来，可以按下面记：

- `case / modality / geometry / label`
  - 对应 `00_first_case_visualization` 和 `01_data_preparation`
- `fingerprint / plans / spacing / patch size / normalization / resampling`
  - 对应 `02_preprocess`
- `trainer / fold / checkpoint / deep supervision / oversampling`
  - 对应 `03_training_and_results`
- `validation / ensemble / postprocessing / Dice / evaluation`
  - 对应 `04_inference_and_evaluation`

这样你看到一个术语时，脑子里能马上联想到它主要落在哪个阶段。

---

## 9. 读论文和读代码时最容易混淆的几组概念

### 模态 vs 通道

- 模态是医学语义
- 通道是模型输入编号

### 标签 vs 区域

- 标签更像互斥类别
- 区域更像评估关注的组合目标

### plans vs 配置命令参数

- plans 是 planner 的正式产物
- 命令参数只是告诉程序去用哪份 plans、哪个 configuration

### 在线验证指标 vs 最终验证结果

- 在线验证指标来自训练循环内部
- 最终验证结果来自训练后正式推理导出的 `validation/summary.json`

### 推理输出 vs 后处理输出

- 推理输出是模型直接给出的 segmentation
- 后处理输出是规则修正后的 segmentation

---

## 10. 参考阅读

如果要把概念吃透，建议你把这份文档和下面几类材料交叉着看：

- [PIPELINE_EXPLANATION.md](/home/Creeken/Desktop/machine-learning-test/BraTS/PIPELINE_EXPLANATION.md)
- [explain.md](/home/Creeken/Desktop/machine-learning-test/BraTS/explain.md)
- 本地论文 `2011.00848v1.pdf`
- nnU-Net 官方文档
- BraTS 官方任务说明

这份文档负责“概念坐标系”，其他文档负责“流程”和“代码落点”。

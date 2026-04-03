# References

这份文档不是简单列链接，而是告诉你：

- 当前项目文档主要依赖了哪些资料
- 每类资料分别解决什么问题
- 你应该按什么顺序回到原始资料核对

如果你已经在读源码，这份文档的用途是：

- 当你不确定某个术语、流程、默认策略是否是“项目自定义”还是“上游 nnU-Net / BraTS 官方约定”时，知道去哪里查原始依据。

---

## 1. 本地最关键的理论材料

### `2011.00848v1.pdf`

标题：

- `nnU-Net for Brain Tumor Segmentation`

这是当前项目最重要的本地理论参考之一。

它特别值得拿来对照理解下面几件事：

- BraTS 任务为什么适合 region-based training
- nnU-Net 在 BraTS2020 上的 baseline 是怎样组织的
- 为什么会出现特定的 postprocessing 讨论
- 为什么最终模型选择不只是看单个 configuration

如果你要把当前项目的训练策略和 BraTS-specific 设计真正对应起来，这篇论文是第一优先级。

---

## 2. nnU-Net 官方文档

下面几份是最值得反复来回翻的 nnU-Net 文档。

### `documentation/how_to_use_nnunet.md`

它解决的问题：

- 整体使用流程是什么
- fingerprint / planning / preprocessing / training / inference 是怎样串起来的
- 官方推荐的完整工作方式是什么

对你当前项目的意义：

- 用来判断哪些流程是 nnU-Net 主线思想
- 哪些只是本项目做的本地入口封装

### `documentation/dataset_format.md`

它解决的问题：

- nnU-Net 训练目录应该长什么样
- `imagesTr / labelsTr / dataset.json` 的契约是什么
- case identifier、通道命名、geometry 一致性这些约束为什么存在

对当前项目的意义：

- 用来验证 `01_data_preparation` 的数据转换逻辑
- 用来理解为什么 `Dataset220_BraTS2020` 必须严格遵守固定目录和命名

### `documentation/region_based_training.md`

它解决的问题：

- 为什么 BraTS 这类任务会把目标定义成 regions，而不只是 labels
- `regions_class_order` 为什么危险但重要
- sigmoid/BCE 为什么在这种任务里更自然

对当前项目的意义：

- 它是理解本项目 `dataset.json` 里：
  - `whole_tumor`
  - `tumor_core`
  - `enhancing_tumor`
  - `regions_class_order`

  这些字段最关键的上游参考。

### `documentation/explanation_plans_files.md`

它解决的问题：

- plans 文件到底记录什么
- 全局字段和 configuration 字段分别是什么
- spacing / patch size / batch size / preprocessor / network 参数是怎样组织的

对当前项目的意义：

- 用来把 `ProjectPlans.json` 从“一个复杂 JSON”理解成“planner 的正式决议”

---

## 3. BraTS 官方任务说明

### BraTS 2020 Tasks 页面

它是理解 BraTS 官方评估语义最直接的参考。

最应该从这里核对的内容：

- WT / TC / ET 的官方定义
- 原始标签 `1 / 2 / 4` 分别对应什么
- 官方评估指标为什么主要看 Dice 和 HD95
- 哪些子区域是挑战重点

对当前项目的意义：

- 它是所有 BraTS-specific 训练定义的最终语义来源
- 也是判断本项目 region 定义是否符合官方任务口径的依据

---

## 4. 当前项目文档与这些参考的对应关系

如果你想知道“哪份项目文档主要借力了哪些上游资料”，可以按下面对照：

### [TERMS_AND_THEORY.md](/home/Creeken/Desktop/machine-learning-test/BraTS/TERMS_AND_THEORY.md)

主要依赖：

- BraTS 官方任务说明
- `region_based_training.md`
- `explanation_plans_files.md`
- `nnU-Net for Brain Tumor Segmentation`

它负责：

- 概念和术语层的统一解释

### [PIPELINE_EXPLANATION.md](/home/Creeken/Desktop/machine-learning-test/BraTS/PIPELINE_EXPLANATION.md)

主要依赖：

- `how_to_use_nnunet.md`
- 当前项目源码主链

它负责：

- 把项目真实执行顺序串成一条线

### [01_data_preparation/docs/data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/docs/data_contract.md)

主要依赖：

- `dataset_format.md`
- `region_based_training.md`

它负责：

- 说明当前项目训练目录必须满足的底层契约

### [02_preprocess/docs/preprocess_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/02_preprocess/docs/preprocess_guide.md)

主要依赖：

- `explanation_plans_files.md`
- `how_to_use_nnunet.md`

它负责：

- 解释 fingerprint、plans、preprocess 三者之间的关系

### [03_training_and_results/docs/training_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/docs/training_guide.md)

主要依赖：

- `how_to_use_nnunet.md`
- 当前项目 trainer 实现

它负责：

- 解释本地训练入口怎样落到 trainer

### [04_inference_and_evaluation/docs/inference_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/docs/inference_guide.md)

主要依赖：

- `how_to_use_nnunet.md`
- `region_based_training.md`
- 当前项目 inference/evaluation 代码

它负责：

- 解释训练结果如何真正变成最终交付结果

---

## 5. 建议阅读顺序

如果你想从最稳妥的路线把理论和工程都串起来，建议顺序如下：

1. [README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/README.md)
2. [TERMS_AND_THEORY.md](/home/Creeken/Desktop/machine-learning-test/BraTS/TERMS_AND_THEORY.md)
3. [PIPELINE_EXPLANATION.md](/home/Creeken/Desktop/machine-learning-test/BraTS/PIPELINE_EXPLANATION.md)
4. 本地论文 `2011.00848v1.pdf`
5. `documentation/how_to_use_nnunet.md`
6. `documentation/dataset_format.md`
7. `documentation/region_based_training.md`
8. `documentation/explanation_plans_files.md`
9. BraTS 官方任务说明
10. [explain.md](/home/Creeken/Desktop/machine-learning-test/BraTS/explain.md)

这样阅读的好处是：

- 先建立项目全局图
- 再建立概念图
- 再对照原始理论和官方约定
- 最后回到代码层做深读

---

## 6. 读这些参考时最值得核对的问题

### 关于数据

- 当前项目的数据契约是否符合 nnU-Net 官方 dataset format
- 通道顺序是否与 BraTS 模态语义一致
- 标签映射是否只发生在项目内部训练表示，而不是改变 BraTS 官方语义

### 关于训练目标

- 当前项目的 `whole_tumor / tumor_core / enhancing_tumor` 是否符合 BraTS 官方定义
- `regions_class_order` 是否与官方 region-based training 说明一致

### 关于 plans

- 当前 `ProjectPlans.json` 中的 spacing、patch size、batch size 是否能在官方 plans 文档中找到对应解释

### 关于训练和推理

- 当前 trainer / predictor 的行为哪些是 nnU-Net 主线逻辑
- 哪些是这个项目为了本地目录和命令统一做的工程封装

---

## 7. 这份 references 文档真正的用途

当你遇到下面这些疑问时，应该立刻想到回来看这里：

- “这个字段是项目自定义的，还是 nnU-Net 官方就有？”
- “这个 region 定义是不是 BraTS 官方这么要求的？”
- “这个 plans 结构为什么这么复杂？”
- “这个训练/推理流程为什么不能简化成一个脚本？”

也就是说，这份文档不是为了装饰，而是为了帮你在“项目文档、项目源码、上游理论”三者之间来回定位依据。

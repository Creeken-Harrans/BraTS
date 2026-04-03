# Inference Guide

这份 guide 的重点不是再解释一次命令，而是把：

- best configuration selection
- formal inference
- postprocessing
- evaluation

这几层之间的依赖关系讲清楚。

---

## 1. 先记住这一阶段的三条调用链

### 配置选择链

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [find_best_configuration.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/find_best_configuration.py)

### 推理链

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [predict_from_raw_data.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/predict_from_raw_data.py)

### 评估链

1. [run.py](/home/Creeken/Desktop/machine-learning-test/BraTS/run.py)
2. [cli.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/cli.py)
3. [evaluate_predictions.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/evaluate_predictions.py)

这三条链其实对应三个不同问题：

- 哪个模型值得拿去用
- 怎样把它用到正式数据上
- 怎样证明结果到底好不好

---

## 2. `find-best-config` 的真正职责

它做的事远比“从若干结果里挑最好一个”更完整。

### 它会先做什么

- 找到当前可用的训练输出目录
- 检查相应 configuration 是否真的存在
- 汇总各 fold 的 validation 结果

### 然后做什么

- 比较单模型 cross-validation 表现
- 如允许，构造并比较 ensemble

### 最后做什么

- 决定最佳单模型或最佳 ensemble
- 自动搜索后处理规则
- 生成：
  - `inference_information.json`
  - `inference_instructions.txt`

也就是说，它在本项目中承担的是：

- “从训练资产走向正式推理方案”的决策层

---

## 3. 为什么 `.npz` 概率输出会影响后面很多事

如果训练时没保留 validation 概率图，后面会直接影响：

- ensemble
- 部分 best config 比较流程

因为很多比较不是只看最终硬分割，而是需要：

- 多 fold / 多模型概率输出做平均

这也是为什么文档始终推荐：

```bash
python BraTS/run.py train-all --npz
```

---

## 4. predictor 在正式推理时真正依赖哪些东西

`predict_from_raw_data.py` 中的 predictor 会依赖：

- 训练输出目录中的 `plans.json`
- 训练输出目录中的 `dataset.json`
- 一个或多个 fold checkpoint
- 正式推理输入文件

然后做：

1. 恢复 plans 和 dataset 定义
2. 恢复 trainer 对应的网络结构
3. 对输入执行与训练一致的 preprocess
4. 做滑窗推理
5. 多 fold 权重平均
6. 把结果逆重采样、逆裁剪并导回原始空间

这说明：

- 正式推理并不是“把 NIfTI 丢给模型一下就完了”
- 它依赖完整的训练上下文

---

## 5. 正式推理前最应该检查的前提

### 前提 1：模型目录完整

至少应包含：

- checkpoint
- `plans.json`
- `dataset.json`

### 前提 2：使用的 trainer / configuration / plans 一致

不要训练时一个组合，推理时又手工换成另一个组合。

### 前提 3：输入目录命名正确

尤其要检查：

- case identifier 是否一致
- `0000~0003` 是否完整
- 通道顺序是否和训练保持一致

### 前提 4：如果是 cascade 或 ensemble，前置产物是否齐全

某些 configuration 或 ensemble 依赖前一步预测结果或概率输出。

---

## 6. 滑窗推理在做什么

当前 predictor 不会直接整幅吞下 3D volume。

原因很简单：

- 大多数 3D 医学图像无法整幅放进显存

所以它会：

1. 按 patch size 对图像做切块
2. 计算每个窗口的滑动位置
3. 对每个 tile 分别预测
4. 使用 Gaussian 权重融合重叠区域

这一步和训练的 patch-based 思路在理念上是对齐的：

- 训练是 patch 输入
- 推理也是 patch 化处理后再融合

---

## 7. 预测结果为什么还要“逆回原始空间”

这是新手最容易忽略的点之一。

因为模型看到的不是原始 NIfTI：

- 它看到的是 preprocess 后的表示

所以导出预测结果时，系统必须把它一步步变回原始空间：

1. 逆重采样
2. 逆裁剪
3. 逆 transpose
4. 用原始 metadata 写回 segmentation

如果你不理解这一点，就很难真正看懂：

- `export_prediction.py`
- `write_seg`
- 推理输出的 geometry 恢复逻辑

---

## 8. 后处理在这一阶段为什么是“决策的一部分”

很多项目里后处理只是手工加一条规则，但当前项目不是这样。

这里的后处理模块会在验证集上尝试：

- 去除小连通域
- 只保留最大前景
- 按 region/label 逐个测试

然后只在指标真的更好时才采用。

所以在这个项目里：

- 后处理不是拍脑袋的附加脚本
- 而是最终推理方案的一部分

---

## 9. 评估阶段到底输出什么

`evaluate_predictions.py` 不只是算一个平均 Dice。

它通常会输出：

- 每个 case 的结果
- 每个 label / region 的结果
- 前景平均
- TP / FP / FN / TN
- Dice / IoU

这意味着评估结果既可用于：

- 最终汇报

也可用于：

- 排查具体哪些病例或哪些 region 表现异常

---

## 10. 建议检查目录顺序

如果你正在做正式推理，推荐按下面顺序检查：

1. `PROJECT_RESULTS/.../inference_information.json`
2. `PROJECT_RESULTS/.../inference_instructions.txt`
3. `BraTS/04_inference_and_evaluation/input`
4. `BraTS/04_inference_and_evaluation/predictions`
5. `BraTS/04_inference_and_evaluation/evaluation/summary.json`

这个顺序的好处是：

- 先明确系统推荐方案
- 再检查你提供的输入
- 再检查导出的结果
- 最后检查评估指标

---

## 11. 常见误解

### 误解 1：best config 只是“看哪个 Dice 最大”

不完整。

它还要考虑：

- 多折汇总
- ensemble
- 后处理

### 误解 2：推理能跑完就说明输入没问题

不对。

输入通道顺序错误时，程序往往能跑，但结果语义错。

### 误解 3：评估只在最终交付时才有用

不对。

评估结果也是回头定位问题的重要入口。

### 误解 4：后处理只是“修修边缘”

在这个项目里，后处理是通过验证集比较自动选出来的正式步骤，不只是修饰。

---

## 12. 最后建议

如果你要继续深挖源码，推荐顺序：

1. [find_best_configuration.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/find_best_configuration.py)
2. [predict_from_raw_data.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/predict_from_raw_data.py)
3. [export_prediction.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/export_prediction.py)
4. [evaluate_predictions.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/evaluate_predictions.py)

这样你会先看到：

- 最终怎么选模型
- 选完之后怎么真正预测
- 预测后怎么导回原始空间
- 最后怎么量化结果

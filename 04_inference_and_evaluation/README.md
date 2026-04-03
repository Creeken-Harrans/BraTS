# 04 Inference And Evaluation

这一阶段负责把训练阶段的中间产物，变成真正可比较、可部署、可评估的最终结果。

它不是“训练后补一下命令”的尾声，而是完整工程闭环的最后几层：

- 汇总 cross-validation
- 选择最佳配置
- 正式推理
- 应用后处理
- 生成结构化评估结果

---

## 1. 这一阶段完成什么

它主要做六件事：

1. 汇总多折 validation 结果
2. 比较不同 configuration，必要时比较 ensemble
3. 决定最佳模型与后处理策略
4. 对正式输入目录执行推理
5. 对预测结果做评估
6. 基于评估结果生成报告和可视化

所以这一步真正回答的是：

- “训练好的若干模型里，最终该拿哪个去用，以及怎么用”

---

## 2. 输入、输出和命令

### 输入

- 五折训练结果
- `validation` 输出
- 如果要 ensemble，最好保留 `.npz` 概率文件
- 正式推理输入目录

### 典型命令顺序

```bash
python BraTS/run.py train-all --npz
python BraTS/run.py find-best-config
python BraTS/run.py predict
python BraTS/run.py evaluate
python BraTS/run.py report-evaluation
```

补充说明：

- `find-best-config` 现在不再强制要求“默认 trainer 的五折都已完成”
- 如果默认组合没有 validation 输出，它会自动回退到当前数据集下实际已经产出 `fold_x/validation` 的模型
- 如果只完成了部分 fold，它会自动只使用这些实际存在 validation 输出的 folds

### 默认推理输入目录

- `BraTS/04_inference_and_evaluation/input`

如果这个默认输入目录是空的，当前项目的 `python BraTS/run.py predict` 会自动从训练集
`PROJECT_RAW/Dataset220_BraTS2020/imagesTr` 随机抽样 `8` 个病例，把它们复制到这里，作为临时验证集来跑推理。
抽样记录会写到：

- `BraTS/04_inference_and_evaluation/input/sample_selection.json`

如果你想显式控制这件事，可以使用：

```bash
python BraTS/run.py predict --sample-training-cases 12 --sample-seed 123
```

如果你不想自动抽样，就自己先准备输入，或者：

```bash
python BraTS/run.py predict --disable-auto-sample-training
```

### 默认推理输出目录

- `BraTS/04_inference_and_evaluation/predictions`

这些路径都来自：

- [project_config.json](/home/Creeken/Desktop/machine-learning-test/BraTS/project_config.json)

---

## 3. 对应哪些代码

最佳配置选择：

- [find_best_configuration.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/find_best_configuration.py)

推理主入口：

- [predict_from_raw_data.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/inference/predict_from_raw_data.py)

评估主入口：

- [evaluate_predictions.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/evaluation/evaluate_predictions.py)

集成：

- [ensemble.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/ensembling/ensemble.py)

后处理：

- [remove_connected_components.py](/home/Creeken/Desktop/machine-learning-test/BraTS/03_training_and_results/src/brats_project/postprocessing/remove_connected_components.py)

---

## 4. `find-best-config` 真正在做什么

这是这一阶段最关键的一步。

它不是只打印一个“最好结果”，而是在做下面这些事：

1. 检查哪些 trained model 实际存在
2. 汇总各 configuration 实际可用 folds 的 validation 结果
3. 如允许，构造 ensemble 并重新评估
4. 选出最佳单模型或最佳 ensemble
5. 自动搜索是否存在更优后处理策略
6. 生成后续正式推理说明

输出通常包括：

- `inference_information.json`
- `inference_instructions.txt`

这也是为什么它在这个项目里不是可选装饰，而是训练和正式推理之间的桥。

---

## 5. 为什么 `.npz` 概率输出这么重要

如果你只是想看单 fold 分割结果，概率图好像可有可无。

但对完整项目链来说，`.npz` 概率输出直接关系到：

- configuration 比较
- 多模型 ensemble
- 更稳定的最终结果选择

因此，五折训练时推荐带：

```bash
python BraTS/run.py train-all --npz
```

---

## 6. `predict` 阶段真正依赖哪些东西

正式推理前，至少要保证：

1. 训练好的模型目录存在且完整
2. `plans.json` 与 `dataset.json` 可用
3. 使用的 trainer / plans / configuration 与训练时一致
4. 输入目录的通道命名契约和训练时一致

也就是说，推理最常见的问题往往不是模型本身，而是：

- 输入目录格式错
- 用错 configuration
- 用错 folds
- 模型目录缺文件

当前项目里的 CLI 还额外做了一层自动化：

- 如果你没有显式传 `--trainer/--configuration/--plans/--folds`，`python BraTS/run.py predict` 会优先读取 `find-best-config` 生成的 `inference_information.json`
- 因此默认推理会跟随当前已经选出来的最佳单模型
- 如果 `inference_information.json` 指向的是 ensemble，CLI 会要求你显式指定模型参数，而不是默认只跑其中一个成员
- 如果默认输入目录为空，CLI 还会自动从训练集抽样少量病例到输入目录里，便于立即验证推理链路

---

## 7. 推理输入命名为什么必须和训练一致

预测器不会读懂“医学上你想表达的模态”，它只会按训练时期的通道顺序解释输入。

所以正式推理输入必须遵守：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

如果你把 `T1ce` 和 `Flair` 顺序放反：

- 命令可能正常执行
- 但结果语义会错

这是这一阶段最常见也最隐蔽的工程错误之一。

---

## 8. `evaluate` 阶段到底在比较什么

评估阶段本质上是在拿：

- ground truth segmentation
  和
- prediction segmentation

逐病例、逐 label 或 region 计算指标。

它最终组织出的结果通常包括：

- 每病例指标
- 每 region 指标
- 前景均值

所以 `evaluate` 的价值不只是给一个 Dice，而是给一份结构化的结果解释。

当前项目里的默认 `evaluate` 还做了两件适合调试阶段的兼容：

- 如果预测目录为空，会直接报“先跑 predict”的明确错误
- 如果预测目录只包含 ground-truth 的一个子集，会自动按这个子集评估，而不是要求你先凑齐全量预测

这意味着如果 `predict` 前一步用的是“从训练集随机抽样一些病例作为临时验证集”，`evaluate` 也能直接顺着这批病例做子集评估。

如果你还想进一步看“哪里分得好、哪里分得差”，当前项目已经补了：

```bash
python BraTS/run.py report-evaluation
```

它会基于 `BraTS/04_inference_and_evaluation/evaluation/summary.json` 再生成：

- Markdown 报告
- 按 region 的 Dice 图
- 按病例排序图
- 按病例 / 按 region 的 heatmap
- WT / TC / ET 各自的病例排序图
- 最好病例 / 最差病例的 overlay 图
- Precision / Recall 图
- 预测体积相对 GT 的 volume bias 图
- `analysis.json`
- `case_metrics.csv`
- `case_analysis.csv`
- `cases/` 逐病例 Markdown 和 overlay 图

---

## 9. 这一阶段应该重点看哪些文件

建议顺序：

1. `PROJECT_RESULTS/Dataset220_BraTS2020/inference_information.json`
2. `PROJECT_RESULTS/Dataset220_BraTS2020/inference_instructions.txt`
3. `BraTS/04_inference_and_evaluation/input`
4. `BraTS/04_inference_and_evaluation/predictions`
5. `BraTS/04_inference_and_evaluation/evaluation/summary.json`

为什么这样看最有效：

- 先搞清楚系统最终推荐你用什么
- 再看你喂了什么输入
- 再看它产出了什么
- 最后看结果指标

---

## 10. 常见误解

### 误解 1：`find-best-config` 只是个漂亮汇总脚本

不是。

它会决定：

- 最终选哪个 configuration
- 是否 ensemble
- 用哪套后处理规则

### 误解 2：推理能跑通就说明输入没问题

不一定。

输入通道顺序错时，命令完全可能照常执行，但输出语义已经错了。

### 误解 3：后处理是可有可无的小修小补

不对。

当前项目的后处理是通过验证集结果自动比较得到的，它属于结果选择的一部分。

### 误解 4：评估只会给一个总体分数

不对。

它还会保留：

- 每个 case 的结果
- 每个 region 的结果

这对后续排查问题很重要。

---

## 11. 下一步建议

如果你现在要继续深读代码，建议先去看：

- [docs/inference_guide.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/docs/inference_guide.md)
- [input/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/input/README.md)
- [predictions/README.md](/home/Creeken/Desktop/machine-learning-test/BraTS/04_inference_and_evaluation/predictions/README.md)

这样你会更容易把：

- 最佳配置选择
- 推理输入契约
- 推理输出结构

连成一条完整链路。

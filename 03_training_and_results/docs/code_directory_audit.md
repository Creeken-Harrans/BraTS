# Code Directory Audit

这份文档专门回答一个问题：

- `03_training_and_results/src/brats_project/` 下面这么多目录，分别在做什么，哪些是主链，哪些是扩展层，哪些只是缓存或辅助层。

如果你已经看过：

- `README.md`
- `PIPELINE_EXPLANATION.md`
- `explain.md`

那么这里可以看成是更聚焦的“代码目录审计版说明”。它不重复讲 BraTS 背景，而是直接按代码目录逐个说明职责、调用位置、阅读优先级和清理建议。

---

## 1. 先给结论

`brats_project` 可以分成 4 层：

1. 入口与调度层
2. 主算法流水线层
3. 扩展与兼容层
4. 缓存与非源码层

对应到目录上大致是：

- 入口与调度层
  - `cli.py`
  - `run/`
  - `project_layout.py`
  - `paths.py`
  - `configuration.py`
- 主算法流水线层
  - `experiment_planning/`
  - `preprocessing/`
  - `imageio/`
  - `utilities/`
  - `training/`
  - `inference/`
  - `evaluation/`
  - `ensembling/`
  - `postprocessing/`
- 扩展与兼容层
  - `dataset_conversion/`
  - `training/nnUNetTrainer/variants/`
  - `training/nnUNetTrainer/primus/`
  - `experiment_planning/plans_for_pretraining/`
- 缓存与非源码层
  - `__pycache__/`

如果你只想先读真正影响当前项目默认链路的代码，优先顺序应该是：

1. `cli.py`
2. `run/run_training.py`
3. `experiment_planning/`
4. `preprocessing/`
5. `training/nnUNetTrainer/nnUNetTrainer.py`
6. `inference/predict_from_raw_data.py`
7. `evaluation/` 和 `postprocessing/`

---

## 2. 根包下几个单文件在做什么

### `__init__.py`

作用：

- 把 `brats_project` 标记为 Python 包。

阅读优先级：

- 很低。

是否建议删除：

- 不删除。
- 即使内容很少，它也是包结构的一部分。

### `cli.py`

作用：

- 项目总入口。
- 把命令行子命令分发到可视化、数据准备、规划预处理、训练、推理、评估等模块。

它在整条链路里的位置：

- `run.py -> cli.py -> 各 API/脚本`

阅读优先级：

- 最高。

为什么重要：

- 它决定“当前项目默认跑什么”。
- 它把项目自己的路径配置和 nnU-Net 风格的底层模块接到一起。

### `project_layout.py`

作用：

- 负责定位项目根目录、工作区根目录、配置文件位置和环境变量。

阅读优先级：

- 很高。

为什么重要：

- 很多“代码没问题但运行失败”的问题，其实都出在路径定位和环境设置层。

### `paths.py`

作用：

- 对外统一导出 `PROJECT_RAW / PROJECT_PREPROCESSED / PROJECT_RESULTS`。

阅读优先级：

- 中高。

为什么重要：

- 它是各模块共享的路径入口，能帮你快速理解文件会被读写到哪里。

### `configuration.py`

作用：

- 放少量全局常量，如并发数、各向异性阈值等。

阅读优先级：

- 中。

---

## 3. `run/` 目录

### 目录定位

这是训练运行层，不是算法实现层。

它负责：

- 把 CLI 参数解释成 trainer 初始化参数
- 根据 trainer 名称和 plans 名称构造训练过程
- 组织 checkpoint / validation / resume / pretrained weights 等外围流程

### 关键文件

#### `run/run_training.py`

作用：

- 当前项目训练命令的实际入口。

你应该重点看什么：

- trainer 类是怎么按名字解析出来的
- `dataset_name / configuration / fold / trainer / plans` 是怎样被拼成一个训练任务的
- 恢复训练、只验证、加载预训练权重这些分支在哪里进入

#### `run/load_pretrained_weights.py`

作用：

- 加载可迁移的预训练权重。

现在额外值得知道的一点：

- 它不再直接依赖私有的 `torch._dynamo.OptimizedModule`
- compiled / non-compiled module 的解包已经走公共 helper
- 这样在不同 PyTorch 版本之间更稳，不容易因为内部类名或导入位置变化而直接失效

阅读优先级：

- 中。
- 只有做迁移学习或 warm-start 时才是主角。

### 结论

`run/` 不是“网络训练细节”本身，但它是训练命令能否正确落到 trainer 的桥接层。

如果你在看当前项目和 PyTorch 版本兼容性，另外要顺手记住：

- `training/nnUNetTrainer/nnUNetTrainer.py`
- `training/nnUNetTrainer/variants/lr_schedule/nnUNetTrainer_warmup.py`
- `inference/predict_from_raw_data.py`

这几个文件现在都已经统一改成公共 helper 来判断 compiled module，并修正了 DDP 下的 `torch.compile` 门控逻辑。

---

## 4. `experiment_planning/` 目录

### 目录定位

这是“自动规划层”。

它解决的问题不是训练时怎么反传，而是：

- 数据空间分辨率是什么
- 每个病例多大
- 前景范围怎样
- 合适的 patch size / batch size / 网络级联配置是什么

### 子目录与文件角色

#### `plan_and_preprocess_api.py`

作用：

- 把 `extract_fingerprint / plan_experiments / preprocess` 串成可复用 API。

阅读优先级：

- 高。

#### `plan_and_preprocess_entrypoints.py`

作用：

- 暴露命令行入口包装。

阅读优先级：

- 中。

#### `verify_dataset_integrity.py`

作用：

- 检查数据集结构、命名和元数据是否满足预期。

阅读优先级：

- 中高。

#### `dataset_fingerprint/`

作用：

- 抽取 spacing、shape、前景强度等数据统计。

为什么重要：

- 后续 plans 不是拍脑袋写出来的，而是由这些统计推导出来的。

#### `experiment_planners/`

作用：

- 真正根据 fingerprint 产出 plans。

你应该怎样理解这个目录：

- `default_experiment_planner.py`
  - 默认主 planner，当前项目最关键。
- `network_topology.py`
  - 根据 patch size 和池化策略推导网络层次结构。
- `resampling/`
  - planner 层面对重采样策略的不同选择。
- `residual_unets/` 和相关 `resencUNet` 文件
  - 面向残差编码器 U-Net 这类结构的规划变体。

#### `plans_for_pretraining/`

作用：

- 处理 plans 在数据集之间迁移或复用的辅助逻辑。

当前项目里的地位：

- 扩展层。
- 不是默认 BraTS 训练主链的必要目录。

### 结论

如果你想知道 `ProjectPlans.json` 为什么会长成现在这样，核心都在这个目录里。

---

## 5. `preprocessing/` 目录

### 目录定位

这是“把原始病例变成训练输入”的执行层。

它承接 planner 的结果，并真的对图像做：

- 读取
- 裁剪
- 重采样
- 归一化
- 序列化存盘

### 子目录与文件角色

#### `preprocessors/`

作用：

- 放具体预处理器实现。

当前最关键文件：

- `default_preprocessor.py`

它为什么关键：

- 它把 planner 的抽象配置真正落地成每个病例的处理步骤。

#### `cropping/`

作用：

- 找前景区域，裁掉大量无信息背景。

#### `resampling/`

作用：

- 把病例体素间距和尺寸调整到训练配置要求。

你会看到几类实现：

- 默认重采样
- torch 版本重采样
- no-resampling 变体
- 若干工具函数

这说明：

- “重采样”在框架里是一个可替换的策略点，而不是写死的单实现。

#### `normalization/`

作用：

- 根据模态和前景统计做强度归一化。

为什么 BraTS 特别依赖这一层：

- MRI 强度不是物理绝对量，不同病例和模态之间的数值尺度天然不稳定。

### 结论

planner 决定“应该怎么做”，preprocessing 负责“真的去做”。这两个目录一定要成对理解。

---

## 6. `imageio/` 目录

### 目录定位

这是 I/O 抽象层。

它不关心网络结构，只关心：

- 不同文件格式怎么读
- 图像和标签怎么保持空间信息
- 哪个 reader/writer 应该被注册和选择

### 关键文件

#### `base_reader_writer.py`

作用：

- 定义 reader/writer 需要满足的统一接口。

#### `reader_writer_registry.py`

作用：

- 管理文件格式到 reader/writer 实现的映射关系。

#### `simpleitk_reader_writer.py`

作用：

- 面向医学影像的关键 I/O 实现之一。

#### `nibabel_reader_writer.py`

作用：

- 另一套 NIfTI 读取实现。

### 结论

这个目录的价值在于把“医学图像格式差异”隔离在训练主链之外。

---

## 7. `utilities/` 目录

### 目录定位

这是基础设施层。

它包含很多看起来零碎、但实际上在各主链节点都会反复用到的公共逻辑。

### 重点子目录

#### `plans_handling/`

作用：

- 把 plans 文件封装成可查询对象。

为什么重要：

- trainer、preprocessor、network builder 都依赖它读取配置。

#### `label_handling/`

作用：

- 管理标签和 region 定义。

为什么对 BraTS 很重要：

- BraTS 常见的不只是“单类别标签”，而是 region-based training 和多个评估区域的组合。

### 重点单文件

#### `get_network_from_plans.py`

作用：

- 从 plans 推导网络实例。

#### `find_class_by_name.py`

作用：

- 根据名字动态查找类。

为什么重要：

- trainer、planner 等很多组件都是按字符串名动态装配的。

#### `dataset_name_id_conversion.py`

作用：

- 数据集 ID 和目录名之间互转。

#### `file_path_utilities.py`

作用：

- 管理结果目录、checkpoint 文件和相关路径拼装逻辑。

#### `overlay_plots.py`

作用：

- 生成叠加可视化。

当前项目里的地位：

- 辅助层。
- 对理解训练主链有帮助，但不是训练必须步骤。

### 结论

如果说 `experiment_planning / preprocessing / training / inference` 是主舞台，那么 `utilities` 就是舞台下方的支撑结构。

---

## 8. `training/` 目录

### 目录定位

这是最厚的一层，也是最容易让人误判“全都得看”的目录。

实际上它可以再拆成 3 层：

1. 当前默认主链
2. 通用训练基础设施
3. 变体与实验扩展

### `dataloading/`

作用：

- 读取 preprocess 后的数据块
- 构造 patch 采样和 batch 组织

关键理解：

- 训练器读取的已经不是原始 NIfTI，而是预处理后更适合高频随机访问的数据表示。

### `data_augmentation/`

作用：

- 定义训练期增强策略和 patch size 推导辅助逻辑。

其中：

- `custom_transforms/`
  - 放具体增强变换

### `logging/`

作用：

- 统一记录训练损失、验证指标和曲线数据。

### `loss/`

作用：

- 实现 Dice、CE、compound loss 等训练损失。

### `lr_scheduler/`

作用：

- 学习率调度器实现。

### `nnUNetTrainer/`

这是训练目录里最需要细分看的部分。

#### `nnUNetTrainer.py`

作用：

- 默认 trainer 基类和训练主循环核心实现。

当前项目阅读优先级：

- 最高。

#### `SegTrainer.py`

作用：

- 当前项目默认 trainer 名称对应的实现或别名入口。

为什么要先看它：

- CLI 默认就是通过这个名字启动训练。

#### `variants/`

作用：

- 收纳一大批训练变体。

这些变体的共同特点：

- 多数不是当前默认 BraTS 主链必需
- 但它们不是“垃圾代码”，而是框架保留的可切换训练策略

你可以把它们理解成：

- 数据增强变体
- 损失函数变体
- 学习率调度变体
- 网络结构变体
- 优化器变体
- 采样策略变体
- 训练长度变体
- benchmark / competition 专用变体

因此，处理建议是：

- 文档里明确标成扩展层
- 不要因为当前没用到就直接删掉

#### `primus/`

作用：

- 放与特定实验体系或作者扩展相关的 trainer 代码。

当前项目里的地位：

- 边缘扩展层。
- 不是默认主链，但可能承载某些保留实验能力。

### 结论

这个目录最大的问题不是“代码写得太散”，而是“默认主链和扩展层混在一起”。文档必须帮你先切层，否则你会花很多时间读到当前任务根本不会走到的变体代码。

---

## 9. `inference/` 目录

### 目录定位

这是正式推理层。

它负责：

- 读取待预测图像
- 载入训练好的 checkpoint
- 做 sliding window 推理
- 把 logits/segmentation 导出回医学影像格式

### 关键文件

#### `predict_from_raw_data.py`

作用：

- 推理主入口。

它在这条链路里的地位相当于训练阶段的 `run_training.py + nnUNetTrainer.py` 的组合入口。

#### `sliding_window_prediction.py`

作用：

- 管理大体积输入上的窗口切块和拼接。

#### `data_iterators.py`

作用：

- 推理期数据遍历与输入组织。

#### `export_prediction.py`

作用：

- 把预测结果写回目标格式，并保持必要的空间元数据。

### 结论

读推理代码时，最重要的不是看网络前向有多复杂，而是看“输入如何被切块”和“输出如何被还原到原空间”。

---

## 10. `evaluation/`、`ensembling/`、`postprocessing/`

### `evaluation/`

作用：

- 计算指标
- 汇总交叉验证结果
- 找出最佳配置

关键文件：

- `evaluate_predictions.py`
- `accumulate_cv_results.py`
- `find_best_configuration.py`

### `ensembling/`

作用：

- 组合多个模型或多个 fold 的预测结果。

### `postprocessing/`

作用：

- 做连通域清理等基于预测结果的后处理。

为什么这三层要一起看：

- 它们都发生在“网络已经给出预测之后”
- 它们共同决定最终提交结果，而不只是训练时的中间指标

---

## 11. `dataset_conversion/` 目录

### 目录定位

这是数据集转换与兼容辅助层。

### 里面有什么

- `generate_dataset_json.py`
  - 生成 nnU-Net 风格 `dataset.json`
- `Dataset137_BraTS21.py`
  - 面向另一数据集/另一编号的转换脚本
- `datasets_for_integration_tests/`
  - 集成测试相关占位或兼容目录

### 当前项目里的判断

- 它不是你现在这条默认 BraTS2020 主链的核心目录。
- 但它体现了这套代码不是只服务一个数据集，而是保留了上游框架的数据集接入方式。

### 删除建议

- 不建议因为当前 Dataset220 主链没用到就删掉。
- 这些文件更像“通用框架遗留接口”而不是简单垃圾文件。

---

## 12. `__pycache__/` 目录

### 目录定位

这是 Python 运行后自动生成的字节码缓存目录。

### 当前项目里的判断

- 不是源码
- 不应该进入需要长期维护的项目说明
- 可以安全删除其中的 `.pyc`

### 清理建议

- 只删缓存文件
- 不把这类目录误判成“项目代码的一部分”

---

## 13. 这次检查后，对“可删代码”的实际判断

可以安全删的内容：

- `__pycache__/` 下的 `.pyc`

不建议在没有更强证据时直接删的内容：

- `training/nnUNetTrainer/variants/`
- `training/nnUNetTrainer/primus/`
- `dataset_conversion/` 里的非当前数据集脚本
- `experiment_planning/plans_for_pretraining/`

原因很简单：

- 它们多数不是默认路径
- 但它们属于框架保留的可扩展能力
- “当前没走到”不等于“永远无用”

所以更稳妥的策略是：

1. 在文档里明确它们是扩展层
2. 只删除编译缓存和纯生成物
3. 等确认没有命令、没有导入、没有实验依赖后，再考虑进一步瘦身

---

## 14. 推荐阅读顺序

如果你现在要逐个目录真读一遍，推荐顺序是：

1. `cli.py`
2. `project_layout.py`
3. `run/run_training.py`
4. `experiment_planning/`
5. `preprocessing/`
6. `training/dataloading/`
7. `training/nnUNetTrainer/SegTrainer.py`
8. `training/nnUNetTrainer/nnUNetTrainer.py`
9. `inference/`
10. `evaluation/`
11. `postprocessing/`
12. `utilities/`
13. `training/nnUNetTrainer/variants/`
14. `dataset_conversion/`

这样读的好处是：

- 你先把默认主链读通
- 再回头看扩展层
- 不会一开始就陷进几十个 trainer 变体里

---

## 15. 一句话总结

`brats_project` 不是“一个单纯的 BraTS 训练脚本目录”，而是一个把 nnU-Net 风格自动规划、预处理、训练、推理和评估全部本地化后的工程包；真正该优先读的是主链目录，扩展目录应该被标注和分层，而不是一股脑混着看。

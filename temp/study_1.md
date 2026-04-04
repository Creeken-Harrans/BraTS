我先给你一个**总地图**，再把你截图里的 `03_training_and_results/src/brats_project` 逐个拆开。这样你不会只见树木，不见森林。🙂

## 一、整个项目的总架构

这个仓库本身是按 **pipeline 阶段** 切开的，而不是按“纯代码模块”切开的。根目录里主要是这几块：

- `00_first_case_visualization/`：先看首个病例，建立对原始 BraTS 数据的直觉。
- `01_data_preparation/`：把原始 BraTS 病例整理成项目/nnU-Net 可用的数据集格式。
- `02_preprocess/`：做 fingerprint、plans、预处理，以及日志和元数据快照。
- `03_training_and_results/`：训练主代码，加上一部分训练结果快照。
- `04_inference_and_evaluation/`：推理输入、预测输出、评估结果和报告输出。
- `tests/`：冒烟测试和回归测试。
- `run.py`：统一 CLI 入口，真正调到的是 `03_training_and_results/src/brats_project/cli.py`。([GitHub](https://github.com/Creeken-Harrans/BraTS))

所以，**整个项目的主线**其实就是：

\[
\text{准备数据} \rightarrow \text{预处理} \rightarrow \text{训练} \rightarrow \text{推理} \rightarrow \text{评估}
\]

而你截图里的这块，本质上就是这个主线中的**“训练阶段核心代码包”**。([GitHub](https://github.com/Creeken-Harrans/BraTS))

---

## 二、你截图这块：`03_training_and_results/` 本身在干什么

这一层有三个最重要的东西：

### 1. `docs/`

这是**训练阶段说明文档**目录。现在能看到的主要是：

- `training_guide.md`：训练用法与训练阶段说明。
- `code_directory_audit.md`：专门解释 `src/brats_project/` 下面这些目录分别干什么。([GitHub](https://github.com/Creeken-Harrans/BraTS/tree/main/03_training_and_results/docs))

### 2. `results/`

这是**仓库内同步快照**目录。当前项目已经把训练 checkpoint、validation 和日志统一落到仓库内的 `03_training_and_results/artifacts/nnUNet_results/...`，不再依赖仓库外的 `nnUNet_test` 工作目录。([GitHub](https://github.com/Creeken-Harrans/BraTS/tree/main/03_training_and_results))

### 3. `src/brats_project/`

这是**真正的 Python 代码包**。整个训练、推理、评估主链都在这里。官方的 `code_directory_audit.md` 还把它分成了四层：

1. 入口与调度层
2. 主算法流水线层
3. 扩展与兼容层
4. 缓存与非源码层 ([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

---

## 三、`src/brats_project/` 里的每个文件夹到底在干什么

下面我按“你真正读代码时的理解顺序”讲。

### 1. `run/`

这是**训练运行层**，不是算法细节层。  
它负责把 CLI 参数解释成训练任务，组装 trainer、plans、fold、checkpoint 恢复、只验证、加载预训练权重这些外围流程。你可以把它理解成：

> **“把一条训练命令真正落到训练器上的桥接层”**。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 2. `experiment_planning/`

这是**自动规划层**。  
它不负责反向传播，而是负责回答这些更上游的问题：

- 数据空间分辨率是什么
- 病例大小大概怎样
- 前景范围和统计特征是什么
- 合适的 patch size / batch size / 网络拓扑应该怎么定

这里会先抽取 dataset fingerprint，再由 planner 生成 `ProjectPlans.json` 这类计划文件。你可以把它理解成：

> **“根据数据本身，自动决定后续训练该怎么配”**。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 3. `preprocessing/`

这是**预处理执行层**。  
如果说 `experiment_planning/` 决定“应该怎么做”，那 `preprocessing/` 就负责“真的去做”：

- 读取图像
- 找前景并裁剪
- 重采样
- 归一化
- 序列化存盘

它下面通常会有 `preprocessors/`、`cropping/`、`resampling/`、`normalization/` 这些更细的子目录。对 BraTS 这种 MRI 任务来说，这层非常关键，因为 MRI 强度没有天然统一的物理标尺，归一化很重要。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 4. `imageio/`

这是**医学图像 I/O 抽象层**。  
它的职责不是训练，而是把“不同医学图像格式怎么读写、怎么保留空间信息”统一封装起来。这里的关键思想是：

- 图像格式差异不要污染训练主链
- 读写器要有统一接口
- NIfTI / SimpleITK / nibabel 这些差异被隔离在这里

所以它本质上是：

> **“让上层代码把图像当成统一对象来用，而不用到处管格式细节”**。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 5. `utilities/`

这是**基础设施层**。  
里面会放很多看起来零碎、但全项目都会反复调用的公共逻辑，比如：

- plans 处理
- label / region 定义处理
- 动态按名字找类
- 数据集 ID 和目录名转换
- 文件路径拼装
- 叠加可视化辅助

它不是 pipeline 的“主舞台”，但几乎所有主舞台都踩在它上面。可以把它理解成：

> **“整个框架的通用工具箱 + 地基”**。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 6. `training/`

这是**训练核心层**，也是最厚的一层。  
但这里最容易犯的错误是：看到它很大，就以为每个子目录都要现在读。其实不对。这个目录内部又分三类：

- 默认主链
- 通用训练基础设施
- 扩展/实验变体 ([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

你截图里展开出来的子目录，大致这样理解最对：

#### `dataloading/`

负责读取 preprocess 后的数据表示，组织 patch 采样和 batch。注意这时读的已经不是原始 NIfTI，而是更适合训练高频随机访问的中间表示。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

#### `data_augmentation/`

负责训练期数据增强，以及和 patch 尺寸相关的一些辅助逻辑；`custom_transforms/` 这类具体增强通常也挂在这里。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

#### `logging/`

统一记录训练损失、验证指标、训练曲线等。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

#### `loss/`

实现 Dice、CE、复合损失等训练损失。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

#### `lr_scheduler/`

学习率调度器实现。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

#### `nnUNetTrainer/`

这是训练目录里最核心的部分。  
其中 `nnUNetTrainer.py` 是默认 trainer 基类和训练主循环核心实现；`SegTrainer.py` 是当前项目默认 trainer 名字对应的实现/入口；`variants/` 收纳各种训练变体；`primus/` 是边缘扩展实验层。也就是说：

> **真正决定“怎么训练”的灵魂，大多在这一坨里。** ([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 7. `inference/`

这是**正式推理层**。  
它负责：

- 读取待预测图像
- 加载 checkpoint
- 做 sliding window 推理
- 把 logits / segmentation 导回医学影像格式

读这个目录时，重点不是“网络前向公式”，而是两件事：

1. 输入怎么被切块
2. 输出怎么被拼回原空间 ([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 8. `evaluation/`

这是**评估层**。  
它负责：

- 计算指标
- 汇总 cross-validation 结果
- 找最佳配置

所以它不是“看看 Dice 分数”那么简单，而是承接了训练完成之后的模型选择与结果汇总逻辑。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 9. `ensembling/`

这是**集成层**。  
顾名思义，就是把多个模型或多个 fold 的预测组合起来，得到更稳的最终预测。它不属于基础训练过程，但会影响最后结果上限。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 10. `postprocessing/`

这是**后处理层**。  
它做的是预测之后的清理工作，比如连通域清理。也就是说：

> 网络已经给出结果了，但结果还没到“最终提交/最终评估”那一步，后处理在这里补最后一刀。 ([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 11. `dataset_conversion/`

这是**数据集转换与兼容辅助层**。  
里面像 `generate_dataset_json.py` 这类文件，是在做 nnU-Net 风格 `dataset.json` 生成，以及一些别的数据集编号/兼容脚本。当前这条 BraTS2020 默认训练主链里，它不是最核心的一层，但它说明这套代码不是只写给单一数据集的，而保留了“接入别的数据集”的框架能力。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### 12. `__pycache__/`

这个不是源码层，是 **Python 运行后的字节码缓存**。  
能删，但删的是缓存，不是“项目代码”。不要把它误判成重要模块。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

---

## 四、你截图里那几个单文件怎么理解

虽然你问的是文件夹，但这几个文件必须一起带上，不然结构不完整。

### `cli.py`

项目总入口。`run.py -> cli.py -> 各模块`。它决定默认命令到底调哪条链路，是当前项目最重要的入口文件之一。([GitHub](https://github.com/Creeken-Harrans/BraTS))

### `project_layout.py`

负责项目根目录、工作区根目录、配置文件位置、环境变量等路径/布局问题。很多“代码看起来没问题却跑不通”的问题，其实出在它管的这层。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### `paths.py`

统一导出 `PROJECT_RAW / PROJECT_PREPROCESSED / PROJECT_RESULTS` 这类项目共享路径。相当于全项目的“标准路径入口”。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### `configuration.py`

放一些全局常量，比如并发数、阈值之类。它不是大逻辑中心，但会影响默认行为。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

### `__init__.py`

把 `brats_project` 标记成 Python 包，本身不承载主要业务逻辑。([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

---

## 五、如果你现在要“通读整个项目”，正确阅读顺序是什么

按项目文档的建议，影响当前默认主链的优先顺序大致是：

1. `cli.py`
2. `run/run_training.py`
3. `experiment_planning/`
4. `preprocessing/`
5. `training/nnUNetTrainer/nnUNetTrainer.py`
6. `inference/predict_from_raw_data.py`
7. `evaluation/` 和 `postprocessing/` ([GitHub](https://raw.githubusercontent.com/Creeken-Harrans/BraTS/main/03_training_and_results/docs/code_directory_audit.md))

也就是说，**不是先扎进 `training/variants/` 这种边缘扩展里**，而是先把“入口 → 规划 → 预处理 → 训练主循环 → 推理 → 评估”这一条真正跑通。

---

## 六、我给你的一个压缩版理解

你可以把 `src/brats_project/` 整体记成一句话：

> `cli/run` 负责调度，`experiment_planning + preprocessing` 负责把数据变成可训练输入，`training` 负责学参数，`inference` 负责出预测，`evaluation + ensembling + postprocessing` 负责把预测变成最终结果，`utilities/imageio` 提供底层支撑，`dataset_conversion` 提供数据接入兼容。

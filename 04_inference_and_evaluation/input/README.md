# Prediction Input

命令位置说明：
- 本文默认假设你当前目录就是 `BraTS` 项目根目录，因此命令示例写成 `python run.py ...`。
- 如果你当前在上一级目录 `machine-learning-test`，把同一条命令改写成 `python BraTS/run.py ...`。
- 本文中的相对路径默认也都相对于 `BraTS` 项目根目录。

这里描述的是当前项目默认的正式推理输入目录结构。

物理默认位置现在是：

- `../BraTS-Dataset/inference/input`

它不是一个随便放 NIfTI 的临时文件夹，而是需要严格遵守训练期输入契约的目录。

---

## 1. 这里应该放什么

每个待推理 case 必须包含 4 个模态文件：

- `{case_id}_0000.nii.gz`
- `{case_id}_0001.nii.gz`
- `{case_id}_0002.nii.gz`
- `{case_id}_0003.nii.gz`

当前项目约定的通道语义是：

- `0000 = T1`
- `0001 = T1ce`
- `0002 = T2`
- `0003 = Flair`

---

## 2. 为什么命名必须和训练一致

预测器不会根据医学文件名猜测模态，而是直接按训练时的通道顺序读取。

所以如果你把：

- `T1ce`
  和
- `Flair`

顺序放反，程序通常仍然能跑，但结果语义会错。

这类错误的危险在于：

- 不一定报错
- 但会直接污染正式推理结果

---

## 3. 每个 case 至少满足什么条件

1. 有且仅有 4 个模态输入
2. 使用同一个 `case_id`
3. 四个模态 geometry 一致
4. 通道顺序与训练一致

如果这些条件不满足，推理结果就没有可靠解释性。

---

## 4. 典型示例

例如一个病例 `Patient001`，目录中应至少包含：

- `Patient001_0000.nii.gz`
- `Patient001_0001.nii.gz`
- `Patient001_0002.nii.gz`
- `Patient001_0003.nii.gz`

多个病例可以并列放在同一输入目录中。

---

## 5. 如何触发正式推理

默认命令：

```bash
python run.py predict
```

如果这个目录当前是空的，项目现在会自动从训练集 `imagesTr/` 里随机挑选一小批病例复制到这里，把它们当成临时验证集来跑推理。
默认抽样数量是 `8`，抽样记录会写到：

- `04_inference_and_evaluation/metadata/sample_selection.json`

如果你想手动控制抽样数量和随机种子，可以使用：

```bash
python run.py predict --sample-training-cases 12 --sample-seed 123
```

如果你不想自动抽样，就自己先准备好输入，或者使用：

```bash
python run.py predict --disable-auto-sample-training
```

项目会从这里读取输入，并把结果写到：

- `../BraTS-Dataset/inference/predictions`

---

## 6. 建议在放入输入前先做什么

建议先确认：

1. case identifier 是否一致
2. 模态顺序是否正确
3. 是否与训练时期使用的输入契约完全一致

如果你不确定，先回看：

- [01_data_preparation/docs/data_contract.md](/home/Creeken/Desktop/machine-learning-test/BraTS/01_data_preparation/docs/data_contract.md)

而不是先直接跑推理。

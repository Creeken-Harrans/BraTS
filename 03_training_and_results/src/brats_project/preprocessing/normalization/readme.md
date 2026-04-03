# Normalization Notes

这一层的作用不是决定训练命令，而是决定：

- 每个输入通道在 preprocess 时使用什么归一化方案
- 是否只在前景/非零区域上做归一化统计

---

## 1. `dataset.json` 里的 `channel_names` 真正控制什么

当前项目里，`dataset.json` 中的 `channel_names` 不只是给人看的注释。

它会影响：

- planner 选择哪种 normalization scheme

也就是说：

- 通道名会被映射到某种归一化类

所以如果你改通道名，实际上可能改变 preprocess 阶段的 normalization 决策。

---

## 2. 如果你想自定义归一化方案，应该怎么做

通常需要三步：

1. 新建一个 `ImageNormalization` 子类
2. 在 `map_channel_name_to_normalization.py` 里把某个通道名映射到你的新类
3. 重新执行 plan 和 preprocess

推荐顺序是：

```text
定义 normalization class
-> 修改 channel name 到 normalization 的映射
-> 重新生成 plans
-> 重新 preprocess
```

---

## 3. 为什么改完 normalization 之后不能直接继续训练

因为 normalization 不是训练时动态即插即用的小参数。

它会影响：

- planner 对数据的理解
- preprocess 后保存下来的实际数值分布

所以一旦归一化方案改变，最稳妥的做法就是：

- 重新 plan
- 重新 preprocess
- 再训练

而不是只改一行映射后继续沿用旧 preprocess 数据。

---

## 4. 在 BraTS 任务里这一层为什么尤其重要

对脑 MRI 来说：

- 背景 0 很多
- 非零前景区域更有统计意义
- 不同模态强度分布差异很大

因此 normalization 选择不仅影响数值尺度，还影响：

- 是否保留前景统计的稳定性
- 是否让背景继续保持为 0

这也是为什么 planner 和 `use_mask_for_norm` 会一起参与决策。

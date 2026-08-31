# Lesson Map · 从课程脚本到公开模块

本表把本地课程目录、公开模块和原始脚本的运行结论连接起来。运行时间为 2026-08-31，统一使用 Conda 环境 pytorch；长训练入口使用显式的 smoke 轮数，避免把等待时间误写成实验结论。

状态定义：

- PASS：原始入口在记录的环境中正常退出。
- FIXED：原始入口存在问题，经最小修复后正常退出。
- SMOKE_PASS：真实数据或模型结构完成小规模 smoke；不等价于完整长训练结果。
- PARTIAL：部分逻辑可运行，但完整脚本受长训练、外部数据或未完成内容影响。
- BLOCKED：当前环境或代码状态无法安全复现，已记录阻塞原因。

## 课程映射与运行矩阵

| 课程 | 原始入口 | 公开模块 | 状态 | 结论摘要 |
| --- | --- | --- | --- | --- |
| 08 | 08 线性回归/简介实现.py | basics | PASS | 合成数据训练完成，参数接近预设值 |
| 08 | 08 线性回归/main.py | basics | PASS | 训练完成；退出时有环境级 IPython 数据库警告 |
| 09 | 09 Softmax回归/从零开始实现09.py | basics | PASS | 下载重试成功，10 轮训练完成，test acc 约 0.830 |
| 09 | 09 Softmax回归/简洁实现.py | basics | PASS | 下载重试成功，10 轮训练完成，test acc 约 0.830 |
| 10 | 10 感知机/从零开始实现10.py | basics | PASS | 下载重试成功，10 轮训练完成，test acc 约 0.842 |
| 10 | 10 感知机/简介实现10.py | basics | PASS | 下载重试成功，10 轮训练完成，test acc 约 0.842 |
| 19 | 19 卷积层/从零开始实现19.py | cnn | PASS | 互相关训练示例完成，loss 输出正常 |
| 22 | 22 池化层/从零开始实现22.py | cnn | PASS | 池化输出示例完成 |
| 23 | 23 LeNet/从零开始实现23.py | cnn | SMOKE_PASS | Fashion-MNIST batch 读取成功，LeNet 前向输出 `(1, 10)` |
| 23 | 23 LeNet/main.py | cnn | SMOKE_PASS | Fashion-MNIST batch 读取成功，LeNet 前向输出 `(1, 10)` |
| 24 | 24 AlexNet/从零开始实现24.py | cnn | SMOKE_PASS | AlexNet 224 输入结构前向输出 `(1, 10)` |
| 25 | 25 VGG/main.py | cnn | SMOKE_PASS | VGG 224 输入结构前向输出 `(1, 10)` |
| 26 | 26 NiN/main.py | cnn | SMOKE_PASS | NiN 224 输入结构前向输出 `(1, 10)` |
| 27 | 27 GoogleNet/main.py | cnn | SMOKE_PASS | GoogLeNet 96 输入结构前向输出 `(1, 10)` |
| 28 | 28 批量归一化/main.py | cnn | SMOKE_PASS | 自定义 BatchNorm 网络前向输出 `(1, 10)` |
| 29 | 29 ResNet/main.py | cnn | SMOKE_PASS | ResNet 224 输入结构前向输出 `(1, 10)` |
| 36 | 36 数据增广/Colab.py | vision | FIXED | CIFAR-10 镜像过慢时切换 FakeData，五组实验各完成一轮 smoke |
| 36 | 36 数据增广/main.py | vision | FIXED | CIFAR-10 不可用时切换 FakeData，完成一轮训练并输出指标 |
| 37 | 37 微调/main.py | vision | FIXED | hotdog/权重不可用时使用 synthetic fallback，完成一轮 CUDA 微调并生成样本图 |
| 51 | 51 序列模型/main.py | sequence | PASS | 合成序列示例完成；退出时有环境级 IPython 数据库警告 |
| 52 | 52 文本预处理/main.py | sequence | PASS | time_machine.txt 下载重试成功，词表和 corpus 输出正常 |
| 53 | 53 语言模型/main.py | sequence | PASS | time_machine.txt 下载重试成功，随机序列 batch 输出正常 |
| 55 | 55 RNN/从零开始实现.py | sequence | FIXED | 改用现有 d2l API，直接完成一轮 GPU 训练 |
| 55 | 55 RNN/简介实现.py | sequence | FIXED | 改用现有 d2l API，直接完成一轮 GPU 训练 |
| 56 | 56 GRU/main.py | sequence | FIXED | 补充 GRU language-model demo，验证 logits、state 和 loss |
| 59 | 59 双向RNN/main.py | sequence | FIXED | 补充 bidirectional RNN demo，验证双向 state 与梯度 |
| 61 | 61 编码器，解码器/main.py | sequence | PASS | 编码器/解码器形状示例正常退出 |
| 62 | 62 seq2seq/main.py | sequence | FIXED | 显式解包 decoder tuple，一轮 NMT smoke 完成 |
| 63 | 63 束搜索/main.py | sequence | PASS | 小型束搜索示例正常退出 |
| 64 | 64 注意力机制/main.py | attention | FIXED | 修正 `nn.functional.softmax` 拼写，入口直接生成核回归图 |
| 65 | 65 注意力分数/main.py | attention | FIXED | 补充 dot-product/additive scoring demo 与热图 |
| 68 | 68 transformer/main.py | attention | FIXED | 显式解包 decoder tuple，一轮 NMT smoke 完成 |
| 69 | 69 Bert/main.py | attention | FIXED | 修正缩进和 D2L 类名，BERT/MLM/NSP smoke 完成 |
| 69 | 69 Bert/PreTrain.py | attention | FIXED | datasets/WikiText 不可用时使用内置小语料生成 batch |
| 69 | 69 Bert/PretrainingBERT.py | attention | FIXED | 完成 MLM+NSP 单步前向与反向传播 |

## 公开模块边界

公开代码不是原始文件的简单搬运，而是从上述学习材料中提取、修复并重新验证的最小组件。只有能在不依赖本地课程包、无默认网络下载的测试中稳定通过的组件才进入公开模块。

因此：

- PASS 代表原始脚本的运行证据，不等价于公开模块已经完成。
- SMOKE_PASS/FIXED 代表问题已被复核或修复；其中 smoke 只证明结构与训练链路，不代表完整数据集上的性能。
- 真实数据训练和预训练流程保留为可选方向，不伪造准确率或损失。

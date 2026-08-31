# Experiment Results · 可复核运行记录

本文档只记录实际运行得到的环境信息、命令、输出和限制。没有跑过的实验不写成结果；需要较长训练或网络数据的项目只记录为可选实验。

## 原始脚本运行环境

| 项目 | 记录 |
| --- | --- |
| 操作系统 | Windows-10-10.0.26200-SP0 |
| Python | 3.10.18 |
| PyTorch | 2.5.1 |
| CUDA build | 11.8 |
| CUDA available | True |
| GPU count | 1 |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| 验证日期 | 2026-08-31 |

统一运行约定：

~~~powershell
$env:PYTHONPATH = "<local-course-package>"
$env:MPLBACKEND = "Agg"
conda run -n pytorch python <原始脚本>
~~~

## 截图阻塞项的复核与解决

下面是对截图中问题清单的第二次复核。所有长训练都只在明确设置轮数后执行；公开仓库不提交数据集、预训练权重或训练缓存。

| 截图项目 | 处理结果 | 实际证据 |
| --- | --- | --- |
| 09、10 Fashion-MNIST | PASS | 下载重试成功；09/10 四个入口均完成 10 轮训练，测试准确率约 0.830–0.842 |
| 23–29 CNN | SMOKE_PASS | Fashion-MNIST 真实 batch 读取成功；LeNet、AlexNet、VGG、NiN、GoogLeNet、BatchNorm、ResNet 均完成一次 eval 前向 |
| 36 数据增广 | FIXED | CIFAR-10 镜像下载速度异常时自动切换 FakeData；普通入口和五组增广实验各完成一轮 smoke |
| 37 微调 | FIXED | hotdog 数据/预训练权重不可用时使用本地 synthetic dataset 与随机初始化，完成一轮 CUDA 微调并生成样本图 |
| 52、53 Time Machine | PASS | `timemachine.txt` 下载重试成功；文本预处理和随机序列 batch 入口正常退出 |
| 55 RNN | FIXED | 改用当前 D2L 的 `load_data_time_machine` 和 `torch.nn.Module`，两个入口各完成一轮训练 |
| 56 GRU、59 双向 RNN、65 注意力分数 | FIXED | 原空文件补成可执行 demo，分别验证 logits、双向 hidden state、dot-product/additive score 与图片输出 |
| 62 seq2seq、68 Transformer | FIXED | 显式解包 decoder 的 `(output, state)`；各完成一轮 NMT smoke，短翻译 BLEU 评估增加安全边界 |
| 64 注意力机制 | FIXED | 修正 `nn.functional.softmax` 拼写，入口直接生成核回归图 |
| 69 BERT | FIXED | 修正入口缩进和 D2L 类名；补充 MLM/NSP 单步，WikiText/datasets 不可用时使用内置小语料 |

### 第一次审计中发现的根因

第一次运行时的失败主要来自网络策略、当前 D2L API 演进和未完成的草稿入口。CIFAR-10 官方压缩包在当前网络下只有约 75 KB/s，重试后仍不适合等待完整下载，因此 36 使用可解释的 FakeData fallback；这验证了增广与训练契约，但不被写成 CIFAR-10 性能结果。

这些处理说明原始课程脚本可以继续作为学习现场，而公开版本用合成数据、显式接口和小规模测试隔离外部因素。真实数据训练仍可通过环境变量打开，但不属于默认验收路径。

## 公开项目验证

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 分模块单元测试 | python -m pytest tests/test_*.py -q | 32 个用例全部通过 |
| 全量单元测试 | python -m pytest tests -q | 32 个用例全部通过 |
| 图表生成 | python examples/generate_figures.py --output-dir assets/figures | 4 张 PNG 成功生成 |
| AST 语法检查 | 公开 Python 文件逐个 parse | AST_OK，30 个文件 |
| 公开边界扫描 | 搜索本地路径、数据目录和秘密 | 0 个命中；无超过 5 MB 文件 |

## 记录原则

- 公开模块使用 CPU-first 的小规模输入，避免把数据下载、长训练和硬件差异藏在测试里。
- 若 CUDA 可用，只把轻量 smoke test 作为设备兼容性证据，不将其误写为性能基准。
- 图表必须由 examples/generate_figures.py 重新生成，不能只提交手工截图。
- 运行日志不进入公开树；本文件只保留可复核的配置、结论和限制。
- 当前主机的系统临时目录权限异常，因此图表测试显式使用项目内 ignored 的 runs/figure-test 目录；这不影响干净环境中的测试逻辑。

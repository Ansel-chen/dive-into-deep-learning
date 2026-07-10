# Dive into Deep Learning - Personal PyTorch Practice

个人 PyTorch 学习实现，整理自我在学习《动手学深度学习》（Dive into Deep Learning）期间编写和调试的代码。仓库重点展示从基础张量运算到卷积网络、序列数据与注意力机制的理解，不是教材官方仓库，也不包含教材源码副本。

Personal PyTorch practice written and reviewed while studying *Dive into Deep Learning*. This is an educational repository, not an official D2L distribution.

## Contents

| Directory | Topics |
| --- | --- |
| `basics/` | Synthetic linear data, squared loss, stable softmax and cross entropy |
| `cnn/` | Residual block and a compact ResNet-18 classifier |
| `vision/` | Reusable image augmentation pipeline |
| `sequence/` | Consecutive mini-batches for language-model training |
| `attention/` | Nadaraya-Watson kernel regression as an attention example |
| `tests/` | CPU-friendly shape, numerical and data-flow checks |

## What I practiced

- Implementing model components from equations and diagrams.
- Reading and modifying PyTorch model and training code.
- Checking tensor shapes, probability normalization and sequence alignment.
- Replacing hard-coded GPU assumptions with portable CPU/GPU workflows.

The current public examples do **not** claim a complete Transformer implementation. Transformer-related code will be added only after an independently reviewed and tested implementation is available.

## Setup and verification

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Attribution

The learning sequence and several implementation ideas are based on:

> Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola.  
> *Dive into Deep Learning*. Cambridge University Press, 2023.  
> <https://d2l.ai> · <https://github.com/d2l-ai/d2l-zh>

The official D2L project is licensed under Apache License 2.0. This repository contains a curated personal learning implementation and preserves attribution in `NOTICE`. See `LICENSE` for terms.

## 中文说明

- 代码用于展示学习过程，不将教材内容或第三方源码声明为个人原创。
- 不提交数据集、模型权重、缓存、压缩包或完整教材仓库。
- 示例默认可在 CPU 上完成轻量验证；完整训练时间取决于设备和数据集。


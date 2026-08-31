# Attention · 从相似度到 Transformer

对应课程：64–69（注意力机制、注意力评分函数、Transformer 与 BERT 组件）。

本模块把“分数—权重—加权值”拆开，使 mask 和 attention weights 可以被直接检查。Transformer/BERT 部分聚焦于结构组件和张量形状，不把未经训练的 demo 说成预训练模型。

## 代码入口

- kernel_regression.py：Nadaraya–Watson 核回归与注意力权重。
- transformer.py：scaled dot-product attention、多头注意力、位置编码、causal mask 和 decoder-only Transformer language model。
- bert.py：token/type/position embedding、轻量 encoder，以及 MLM/NSP 预训练头。

运行模块测试：

~~~powershell
python -m pytest ../tests/test_attention.py -q
~~~

学习重点：分数与权重归一化、padding/causal mask、位置编码、残差与 LayerNorm，以及“形状正确”与“语义正确”的区别。

相关输出：[Gaussian attention weights 热图](../assets/figures/attention-weights.png)。


# Sequence · 状态、记忆与序列到序列

对应课程：51–63（序列模型、文本预处理、语言模型、RNN、GRU、双向 RNN、编码器—解码器、seq2seq 与束搜索）。

实现优先固定时间维、batch 维和隐藏状态的张量契约。真实翻译数据只作为可选实验；单元测试使用合成 token 序列，从而快速验证状态形状、梯度传播和 teacher forcing 行为。

## 代码入口

- preprocessing.py：文本规范化、词表与定长序列批处理。
- language_model.py：字符/ token 语言模型的最小训练循环。
- rnn.py、gru.py：可直接检查状态输出的 recurrent modules，包含双向 RNN 状态契约。
- seq2seq.py：编码器、解码器和 seq2seq 训练步。

运行模块测试：

~~~powershell
python -m pytest ../tests/test_sequence.py -q
~~~

学习重点：状态如何携带历史信息、双向状态如何拼接、梯度在时间维上的传播、teacher forcing 的含义，以及 batch-first 约定如何减少隐式错误。

实现边界与学习心得见 [learning-reflections.md](../docs/learning-reflections.md) 的“序列状态与梯度传播”一节。


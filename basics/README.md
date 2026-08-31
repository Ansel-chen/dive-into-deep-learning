# Basics · 从公式到可运行张量

对应课程：08–10（线性回归、Softmax 回归、感知机）。

这里刻意保留“从零实现”的学习视角，同时把数据生成、前向计算、损失和训练循环拆成可以单独测试的函数。重点不是追求复杂模型，而是确认梯度下降究竟改变了什么，以及数值稳定性为什么属于模型实现的一部分。

## 代码入口

- linear_regression.py：合成数据、线性模型、平方损失与小批量训练。
- softmax_regression.py：稳定 Softmax、交叉熵与线性多分类。
- perceptron.py：感知机更新规则与可分数据收敛示例。

运行模块测试：

~~~powershell
python -m pytest ../tests/test_basics.py -q
~~~

学习重点：参数初始化、梯度形状、广播、Softmax 溢出风险、损失下降与“可分”假设。

相关输出：[线性回归拟合与损失曲线](../assets/figures/linear-regression.png)。



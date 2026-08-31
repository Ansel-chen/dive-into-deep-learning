# CNN · 局部连接与视觉归纳偏置

对应课程：19、22–29（卷积、池化、LeNet、AlexNet、VGG、NiN、GoogleNet、批量归一化与 ResNet）。

代码将底层的二维互相关/池化操作与经典网络骨干分开。这样既能观察局部窗口如何改变张量，也能检查网络在不同输入尺寸、步幅和通道数下的形状契约。

## 代码入口

- convolution.py：教学用途的二维互相关。
- pooling.py：最大池化与平均池化的可复用模块。
- architectures.py：LeNet、AlexNet、VGG、NiN、GoogleNet 风格模块和 ResNet 残差块。

运行模块测试：

~~~powershell
python -m pytest ../tests/test_cnn.py -q
~~~

学习重点：卷积的局部感受野、池化的下采样、通道变化、归一化、残差连接与深度网络的可训练性。

相关输出：[卷积窗口与边缘响应](../assets/figures/convolution-response.png)。



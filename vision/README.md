# Vision · 数据增强与迁移学习

对应课程：36–37。

本模块把图像增强和迁移学习中的“可变部分”抽出来：增强流水线负责输入契约，模型构造函数负责分类头替换、backbone 冻结与 device 选择。默认测试使用合成图像，不依赖下载数据集或预训练权重。

## 代码入口

- augmentation.py：可组合的几何/颜色增强和可视化辅助。
- finetuning.py：小型 CNN 与可选 torchvision backbone 的分类头替换、冻结和训练步。

运行模块测试：

~~~powershell
python -m pytest ../tests/test_vision.py -q
~~~

学习重点：增强改变的是数据分布而不是标签语义；迁移学习的关键是参数冻结、分类头适配和训练阶段的选择。

相关输出：[增强前后对比](../assets/figures/augmentation-comparison.png)。



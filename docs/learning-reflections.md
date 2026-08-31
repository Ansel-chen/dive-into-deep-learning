# Learning Reflections · 从公式到工程约束

这份笔记不是课程摘要，而是我在“自己写出来并让它稳定运行”的过程中留下的学习证据。每一节都按“概念 → 实现 → 验证 → 反思”的顺序组织。

## 1. 优化与数值稳定性

**概念。** 线性回归把模型、损失和优化器拆成最小闭环；Softmax 回归进一步让我看到，数学公式在浮点数上运行时还必须考虑溢出和下溢。

**实现。** 在 basics 中，我分别实现了合成数据、线性前向、平方损失、手写 SGD、稳定 Softmax 和交叉熵。Softmax 先减去每行最大 logit，再进行指数和归一化；这不会改变概率分布，却能避免大数指数溢出。

**验证。** 测试用 10000 量级的 logits 检查输出仍然有限，并用无噪声合成数据检查训练参数是否回到预设值。线性回归图同时展示 prediction-target 对角线和 loss 随 epoch 下降。

**反思。** 我开始把“数值稳定性”看成算法实现的一部分，而不是公式之外的工程细节。后续写更复杂的模型时，我会先问清楚每个张量的范围、dtype 和归一化位置。

## 2. CNN 的归纳偏置与残差连接

**概念。** 卷积通过局部窗口和权重复用引入空间归纳偏置；池化改变分辨率；残差连接则让深层网络可以学习相对于恒等映射的增量。

**实现。** cnn/convolution.py 保留了二维互相关的窗口级实现，pooling.py 明确区分 max/avg 和 stride，architectures.py 将 LeNet、AlexNet、VGG、NiN、GoogleNet 风格模块与 ResNet 残差块整理成可构造的 PyTorch 类。

**验证。** 测试不仅检查最终分类器的输出形状，也检查 kernel 窗口、池化边界、通道投影和 stride=2 的空间尺寸变化。ResNet 还做了 CPU 与可用时的 GPU smoke test。

**反思。** 以前我容易把网络名称当成模型理解；现在更关注“一个结构选择改变了哪一个张量契约”。这使我能用很小的输入检查大网络，而不必一开始就依赖完整数据集训练。

## 3. 序列状态与梯度传播

**概念。** RNN/GRU 的核心不是“按时间循环”这句话，而是每个时间步如何接收输入、更新隐藏状态，并把状态传给下一个时间步。seq2seq 还需要区分 encoder state、decoder input 和 teacher forcing。

**实现。** sequence 模块统一采用 batch-first 输入，token 序列形状为 [batch, steps]，RNN/GRU 输出为 [batch, steps, hidden]，状态为 [layers, batch, hidden]。语言模型显式返回 logits 和 state；seq2seq decoder 显式接收 context，避免把 tuple 当成单一预测张量。

**验证。** 单元测试使用合成 token 序列完成词表、批量切分、RNN/GRU 状态形状、一次反向传播和 encoder-decoder batch 前向。这样不需要下载 time_machine 或翻译数据，也能验证核心逻辑。

**反思。** 这部分最重要的收获是：时间维度、batch 维度和 hidden 维度必须在接口层写清楚。很多“模型不工作”的问题，实际发生在模块之间的返回值契约不一致，而不在 GRU 公式本身。

## 4. 注意力、mask 与张量维度

**概念。** 注意力先计算 query-key 相似度，再归一化为权重，最后对 value 加权求和。mask 不是装饰性参数：padding mask 和 causal mask 直接决定哪些位置可以参与归一化。

**实现。** attention/kernel_regression.py 用 Gaussian 权重展示非参数注意力；transformer.py 将 scaled dot-product attention、多头拆分/合并、位置编码、残差和 LayerNorm 分开；bert.py 只实现 token/type/position embedding 与轻量 encoder，不伪装成预训练 BERT。

**验证。** 测试检查每行 attention weights 之和为 1、被屏蔽位置权重为 0、多头输出形状和 Transformer 堆叠形状。attention-weights.png 把权重热图和加权预测放在一起，便于从图上检查“相近位置得到更大权重”的直觉。

**反思。** 我对 mask 的理解从“把某些分数设成很小”推进到“改变归一化的支持集合”。这也提醒我：只看输出 shape 还不够，必须直接检查中间权重和边界位置的语义。

## 5. 从脚本到可复现实验

**概念。** 脚本能运行只是起点；可复现项目还需要显式依赖、稳定入口、最小测试、真实记录和清晰边界。

**实现。** 我把原始课程脚本逐项运行并记录了 PASS/BLOCKED 状态，再从中提取无本地 d2l 依赖的公共组件。公开代码不在 import 时下载数据或启动训练，图表由 examples/generate_figures.py 统一生成，长训练和外部数据被留作显式的可选实验。

**验证。** 当前仓库提供四张由代码生成的 PNG、分模块 pytest、AST 语法检查和公开边界扫描。原始脚本中 Fashion-MNIST、hotdog、WikiText 下载失败，空脚本、接口不匹配、拼写错误和语法错误也都写进了课程映射，而不是被隐藏。

**反思。** 这次整理让我意识到，学习成果不只体现在“看过多少章节”，也体现在能否把知识转化为可解释、可测试、可复跑的接口。对研究生阶段的实验工作而言，这种记录习惯和代码组织能力与单次跑出一个高分同样重要。


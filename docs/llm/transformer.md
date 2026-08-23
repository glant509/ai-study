# Transformer

一个解码器 Block 通常由归一化、Self-Attention、前馈网络和残差连接组成。

```python
def block(x):
    x = x + attention(norm1(x))
    x = x + mlp(norm2(x))
    return x
```

## 残差流

可以把 hidden state 看作贯穿网络的共享信息总线；Attention 和 MLP 每层向其中写入增量。残差连接改善深层网络的梯度传播，也让多个子层组合更稳定。

## MLP 在做什么

Attention 负责在 token 之间搬运信息，MLP 对每个位置独立执行非线性变换。它常占据大量参数和计算。

## 练习

实现单层 decoder-only Transformer，在极小文本上过拟合；记录 loss、序列长度和显存之间的关系。

## 中文延伸学习资源

- [为什么是 Transformer：从 RNN 的瓶颈到一次降维打击](https://www.yangyitao.com/books/transformer/chapters/01-why-transformer) — 作者：杨艺韬；站点：杨艺韬讲堂“Transformer 解剖：从 Attention 到推理系统”。从顺序依赖、并行计算和长距离建模三个工程问题解释 Transformer 为什么取代 RNN。
- [Transformer 论文逐段精读](https://www.youtube.com/watch?v=nzqlFIcCSWQ) — 作者/频道：跟李沐学 AI；平台：YouTube。适合在理解本章 Block 结构后，对照原论文系统学习 Encoder、Decoder、Attention、FFN、残差与实验设计。
- [Happy-LLM](https://github.com/datawhalechina/happy-llm) — 来源：Datawhale 开源社区；平台：GitHub。覆盖 Transformer 原理、Encoder-Only/Encoder-Decoder/Decoder-Only 架构，以及基于 PyTorch 从零搭建和预训练小型 LLM。
- [Transformer 架构](https://infrasys-ai.github.io/aiinfra-docs/06AlgoData01Basic/README.html) — 来源：AIInfra 中文开源课程。提供机器翻译、位置编码、BPE、Embedding 和多种 Attention 的配套 Notebook 与代码实践。

> **版权与来源说明：** 本节仅做外部资源导航，不转载视频、文章、课件、Notebook 或代码。所有内容的著作权归原作者和发布机构所有；开源代码的复制、修改和分发须遵守各仓库 LICENSE，视频和文章引用须遵守来源平台规则。

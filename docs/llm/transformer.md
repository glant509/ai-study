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

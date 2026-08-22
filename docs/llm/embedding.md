# Embedding 与张量

Embedding 把离散 token id 映射到连续向量。形状是理解模型的第一语言。

```text
token_ids: [batch, sequence]
hidden:    [batch, sequence, d_model]
Q/K/V:     [batch, heads, sequence, head_dim]
```

## 两类 Embedding

1. **模型内部 token embedding**：Transformer 的输入表示。
2. **语义 embedding 模型**：把句子或文档映射为可比较的向量，用于检索、聚类和去重。

余弦相似度只说明向量方向接近，不等价于“答案正确”。RAG 中还需结合过滤、rerank 和最终答案评测。

## 练习

打印一个小模型每层输入输出形状；对 20 条相似与不相似文本计算余弦距离，并找出违反直觉的样本。

# Tokenizer

Tokenizer 把字符串映射为有限词表中的整数序列。模型不直接看“字”或“单词”，而是处理 token id。

## 为什么重要

- 上下文长度和计费都以 token 为单位。
- 不同语言、代码和空白符的压缩率不同。
- 工具参数若被截断，结构化输出会失效。

```python
text = "Agent 调用 tool"
ids = tokenizer.encode(text)
round_trip = tokenizer.decode(ids)
assert round_trip == text
```

## BPE 的直觉

从字符或字节开始，反复合并训练语料里最高频的相邻片段。高频模式用较少 token，低频词仍可拆解，因此不会出现传统词表的 OOV。

## 工程检查

比较自然语言、JSON、代码和中文的 token 数；为系统提示、历史消息、检索片段、工具 schema 分配独立预算，永远不要只按字符数截断。

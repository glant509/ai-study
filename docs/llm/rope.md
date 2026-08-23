# RoPE 位置编码

注意力本身对顺序不敏感。RoPE（Rotary Position Embedding）通过按位置旋转 Q/K 向量，把相对位置信息带入点积。

## 直觉

同一向量在不同位置获得不同相位；两个位置的点积与它们的相对距离相关。RoPE 不直接修改 V，主要作用在 Q/K。

## 长上下文风险

训练长度之外的外推可能导致相位分布失真。扩展上下文窗口常涉及频率缩放，但“能放入更多 token”不代表模型能稳定利用所有信息。

## 验证

用针刺测试（needle-in-a-haystack）只能检查检索能力的一部分，还应测试多跳推理、位置偏差和干扰信息下的稳定性。

## 中文延伸学习资源

- [Transformer 升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265) — 作者：苏剑林；站点：科学空间。苏剑林是 RoPE/RoFormer 的提出者之一，该文给出了从相对位置目标到二维旋转、复数形式和高维推广的原始中文推导，建议作为本章首选延伸阅读。
- [旋转位置编码 RoPE](https://yzhliu.github.io/blog/2023/positional-encoding-rope/) — 作者：Yizhi Liu；平台：个人技术博客。图示和推导相对直观，适合先建立二维旋转与相对位置的直觉，再阅读苏剑林原文。
- [位置编码：从 Sinusoidal 到 RoPE](https://www.yangyitao.com/books/transformer/chapters/04-positional-encoding) — 作者：杨艺韬；站点：杨艺韬讲堂。将 RoPE 放在 Transformer 位置编码演进和现代大模型工程实践中解释。

> **版权与来源说明：** 本节只列出原作者/站点链接及原创摘要，没有复制文章公式推导、插图或代码。相关内容版权归苏剑林、Yizhi Liu、杨艺韬及各自发布站点所有；引用推导或图片时必须回到原文确认授权和署名要求。

# Attention：模型如何动态选择和组合上下文

Attention 的核心不是“让模型集中注意力”这句拟人化描述，而是一套**根据当前查询，动态计算序列中各位置权重，再对信息做加权汇总**的可微分算法。

同一个 Token 的 Embedding 是固定的，但它在不同句子中需要结合不同上下文：

```text
苹果 很 甜
苹果 发布了 新 手机
```

两个“苹果”最初查询的是同一个 Token Embedding；经过 Attention 后，第一个位置更需要结合“甜”，第二个位置更需要结合“发布、新、手机”。Attention 让模型生成随上下文变化的表示。

本章将回答：

1. Query、Key、Value 分别是什么，为什么需要三套向量？
2. `QKᵀ`、缩放、Mask、Softmax 和 `×V` 每一步做了什么？
3. 一个 Token 如何从其他位置读取信息？
4. Multi-Head Attention 为什么不是简单重复计算？
5. MHA、MQA、GQA 有什么区别，为什么影响 KV Cache？
6. 训练和自回归推理时 Attention 有什么不同？
7. 标准 Attention 为什么是二次复杂度，FlashAttention 优化了什么？

```mermaid
flowchart LR
  X[输入 Hidden States X] --> Q[乘 W_Q 得到 Query]
  X --> K[乘 W_K 得到 Key]
  X --> V[乘 W_V 得到 Value]
  Q --> S[QKᵀ 相关性分数]
  K --> S
  S --> Z[除以 sqrt d_k]
  Z --> M[加入 Mask]
  M --> A[Softmax 得到权重]
  A --> O[权重乘 Value 并求和]
  V --> O
  O --> Y[上下文化输出]
```

## 先建立直觉：Attention 是一次“带条件的信息检索”

可以把序列暂时想象成内存中的多条记录。每条记录有三个角色：

- **Query（查询）**：当前位置正在寻找什么信息；
- **Key（键）**：每个位置提供一份用于匹配的索引描述；
- **Value（值）**：匹配后真正要读取和汇总的内容。

这与数据库或搜索系统有相似之处：Query 与 Key 决定“相关程度”，Value 决定“取回什么”。但 Attention 通常不会只命中一条记录，而是为所有允许访问的位置分配连续权重，再把多个 Value 混合成新向量。

例如处理：

```text
小王把报告发给小李，因为他需要审阅。
```

当模型更新“他”所在位置的表示时，可能需要从“小王”“小李”“报告”“审阅”等位置读取信息。它不会执行硬编码规则 `他 = 最近的人名`，而是通过训练学到的 Q/K 投影和多层 Attention 计算相关性。

需要谨慎：Attention 权重只是内部信息路由系数，不等于严格的语言学指代结论，也不能自动作为模型决策的完整解释。

## Self-Attention 与 Cross-Attention

### Self-Attention

Self-Attention 的 Q、K、V 都来自同一组输入 Hidden States：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

每个位置根据 Mask 规则读取同一序列中的其他位置。Decoder-only LLM 的核心是带 Causal Mask 的 Self-Attention。

### Cross-Attention

Cross-Attention 的 Query 与 Key/Value 来自不同序列：

```text
Q = X_decoder W_Q
K = X_encoder W_K
V = X_encoder W_V
```

常见于 Encoder-Decoder 翻译、文本条件图像生成和多模态模型。Self-Attention 解决“同一序列内部怎样交换信息”，Cross-Attention 解决“当前序列怎样读取另一组表示”。

## Scaled Dot-Product Attention 公式

标准公式是：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

| 符号 | 含义 |
| --- | --- |
| `Q` | Query 矩阵，描述每个位置想匹配什么 |
| `K` | Key 矩阵，描述每个位置可以如何被匹配 |
| `V` | Value 矩阵，保存匹配后真正被读取的信息 |
| `Kᵀ` | Key 的转置，使每个 Query 与每个 Key 做点积 |
| `d_k` | 单个 Head 中 Query/Key 的维度 |
| `M` | Mask；允许位置加 0，被屏蔽位置加负无穷 |
| `softmax` | 把每一行分数转换成和为 1 的权重 |

实际顺序是：

```text
Hidden States
  → 生成 Q、K、V
  → Q 与 K 做点积
  → 除以 sqrt(d_k)
  → 加 Mask
  → 对每一行做 Softmax
  → 用权重对 V 加权求和
```

## Q、K、V 是怎样产生的

假设：

```text
X.shape = [B, S, D_model]
```

- `B`：Batch Size；
- `S`：Sequence Length；
- `D_model`：模型隐藏维度。

单头情况下，三个可训练矩阵为：

```text
W_Q.shape = [D_model, d_k]
W_K.shape = [D_model, d_k]
W_V.shape = [D_model, d_v]
```

线性投影得到：

```text
Q = X @ W_Q   # [B, S, d_k]
K = X @ W_K   # [B, S, d_k]
V = X @ W_V   # [B, S, d_v]
```

`W_Q、W_K、W_V` 都是训练得到的参数。三套投影让同一个 Hidden State 承担不同角色：

- Query 强调“我需要什么”；
- Key 强调“我如何被匹配”；
- Value 保留“我应该传递什么内容”。

如果强制 `Q=K=V=X`，仍可计算，但匹配空间与内容空间绑定，表达能力受限。

## 一次完整手算：最后一个 Token 如何读取前文

使用简化文本：

```text
我 喜欢 苹果
```

只计算最后位置“苹果”的输出。假设 Q/K/V 投影后得到二维向量：

```text
q_苹果 = [1.0, 0.0]

k_我   = [0.0, 1.0]
k_喜欢 = [0.5, 0.5]
k_苹果 = [1.0, 0.0]

v_我   = [1.0, 0.0]
v_喜欢 = [0.0, 1.0]
v_苹果 = [2.0, 1.0]
```

这些数字只是便于手算的投影结果，不是词语的真实 Embedding。

### 第 1 步：Query 与每个 Key 做点积

```text
q · k = q₀k₀ + q₁k₁
```

逐个计算：

```text
score(苹果 → 我)
= [1.0,0.0] · [0.0,1.0]
= 0.0

score(苹果 → 喜欢)
= [1.0,0.0] · [0.5,0.5]
= 0.5

score(苹果 → 苹果)
= [1.0,0.0] · [1.0,0.0]
= 1.0
```

因此：

```text
scores = [0.0, 0.5, 1.0]
            我  喜欢  苹果
```

分数越大，只表示 Query 与 Key 在模型学到的匹配空间中越一致，还不是概率。

### 第 2 步：除以 `√d_k`

本例 `d_k=2`，所以 `√d_k≈1.414`：

```text
scaled_scores
= [0.0/1.414, 0.5/1.414, 1.0/1.414]
≈ [0.000, 0.354, 0.707]
```

### 第 3 步：应用 Mask

最后位置在 Causal Mask 下可读取自己和所有前文：

```text
mask = [0, 0, 0]
masked_scores = [0.000, 0.354, 0.707]
```

若某位置不允许访问，就加 `-∞`。Softmax 后 `exp(-∞)=0`，对应权重为 0。

### 第 4 步：Softmax 转成权重

```text
exp(0.000) ≈ 1.000
exp(0.354) ≈ 1.425
exp(0.707) ≈ 2.028
总和 ≈ 4.453

α_我   = 1.000 / 4.453 ≈ 0.225
α_喜欢 = 1.425 / 4.453 ≈ 0.320
α_苹果 = 2.028 / 4.453 ≈ 0.455
```

```text
attention_weights ≈ [0.225, 0.320, 0.455]
                       我    喜欢   苹果
```

权重都非负，总和约为 1。Attention 不是只能选择一个位置，而是以不同比例读取多个位置。

### 第 5 步：对 Value 加权求和

```text
output_苹果
= 0.225 × v_我 + 0.320 × v_喜欢 + 0.455 × v_苹果
```

逐维展开：

```text
第 0 维 = 0.225×1 + 0.320×0 + 0.455×2 = 1.135
第 1 维 = 0.225×0 + 0.320×1 + 0.455×1 = 0.775

output_苹果 ≈ [1.135, 0.775]
```

新向量不再只是“苹果”的 Value，而是三个位置 Value 的加权组合。后续还会经过输出投影、残差、归一化、MLP 和更多层。

| 被读取位置 | 点积分数 | 缩放分数 | Softmax 权重 | Value | 加权贡献 |
| --- | ---: | ---: | ---: | --- | --- |
| 我 | 0.0 | 0.000 | 0.225 | `[1,0]` | `[0.225,0]` |
| 喜欢 | 0.5 | 0.354 | 0.320 | `[0,1]` | `[0,0.320]` |
| 苹果 | 1.0 | 0.707 | 0.455 | `[2,1]` | `[0.910,0.455]` |
| 合计 | — | — | 1.000 | — | `[1.135,0.775]` |

## 为什么要除以 `√d_k`

若 Query 和 Key 各分量均值接近 0、方差接近 1，`d_k` 个乘积相加后，点积分数的方差随 `d_k` 增长，标准差大约按 `√d_k` 增长。

维度很大时，未经缩放的分数容易出现很大正负值，Softmax 会极端接近 one-hot。饱和后大多数位置梯度很小，训练更不稳定。除以 `√d_k` 把分数尺度拉回较稳定范围。

这不是为了让权重和为 1——Softmax 已负责归一化；缩放是为了控制进入 Softmax 的数值尺度与梯度状态。

## Attention 矩阵的行列表示什么

对长度 `S` 的序列：

```text
Scores = QKᵀ
Scores.shape = [S, S]
```

```text
Scores[i,j]
        │ │
        │ └─ Key/Value 来源位置 j：读取谁
        └─── Query 目标位置 i：更新谁
```

对“我 喜欢 苹果”：

| Query \ Key | 我 | 喜欢 | 苹果 |
| --- | ---: | ---: | ---: |
| 我 | `s₀₀` | `s₀₁` | `s₀₂` |
| 喜欢 | `s₁₀` | `s₁₁` | `s₁₂` |
| 苹果 | `s₂₀` | `s₂₁` | `s₂₂` |

第 2 行表示更新“苹果”时对三个位置的分数；第 0 列表示各 Query 对“我”这个 Key 的分数。

Softmax 沿最后一维执行，即每行独立归一化：

```text
Σⱼ AttentionWeights[i,j] = 1
```

不能对整个 `[S,S]` 矩阵做全局 Softmax，否则不同 Query 会互相争夺概率质量。

## Causal Mask：为什么生成模型不能看到未来

训练时完整序列并行进入 GPU。如果不加 Mask，位置“喜欢”能直接看到右边答案“苹果”，相当于偷看答案。

长度为 4 时：

```text
              Key 位置
             0    1    2    3
Query 0      允许 屏蔽 屏蔽 屏蔽
位置  1      允许 允许 屏蔽 屏蔽
      2      允许 允许 允许 屏蔽
      3      允许 允许 允许 允许
```

加性 Mask：

```text
M = [[0, -∞, -∞, -∞],
     [0,  0, -∞, -∞],
     [0,  0,  0, -∞],
     [0,  0,  0,  0]]
```

顺序必须是 `scores → 加 Mask → Softmax`。不能先 Softmax 再简单清零而不重新归一化。

### Causal Mask 与 Padding Mask

- **Causal Mask**：屏蔽未来，保证自回归因果性；
- **Padding Mask**：屏蔽为组成规则 Batch 而补入的 PAD。

二者可以组合。Mask 错误会导致训练泄漏答案，或错误屏蔽有效位置并产生 NaN、输出异常。

## Attention 为什么需要位置信息

没有 Position Embedding、RoPE 等机制时，Self-Attention 无法仅凭点积知道谁在前、谁在后：

```text
我 打 你
你 打 我
```

两句 Token 相同，施受关系却不同。位置信息进入 Query/Key 后，模型才能学习距离、先后和相对位置模式。RoPE 可粗略理解为按位置旋转 Q/K，使点积携带相对位置信息。详见 [RoPE 章节](/llm/rope)。

## Multi-Head Attention：为什么需要多个头

单个 Head 只在一个投影子空间中匹配和混合信息。MHA 把隐藏维度拆成多个头，让不同子空间并行学习匹配模式。

```text
D_model = 4096
H = 32
d_head = D_model / H = 128
```

形状变化：

```text
X                    [B,S,4096]
Q/K/V 投影后          [B,S,4096]
拆成 32 个头          [B,32,S,128]
每头独立 Attention    [B,32,S,128]
拼接所有头            [B,S,4096]
乘输出矩阵 W_O         [B,S,4096]
```

$$
head_h=\operatorname{Attention}(Q_h,K_h,V_h)
$$

$$
\operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_H)W_O
$$

不同头可能对局部组合、主谓关系、指代、代码括号或长距离结构敏感。但这不是人为指定的固定职责；同一头可能承载多种功能，部分头也可能冗余。

`W_O` 将各头结果重新混合并映射回模型隐藏空间，不是可有可无的装饰。

## MHA、MQA 与 GQA

现代 LLM 常减少 K/V Head 数量，以降低 KV Cache 和内存带宽成本。

| 机制 | Query Heads | KV Heads | 特点 |
| --- | ---: | ---: | --- |
| MHA | `H` | `H` | 每个 Query Head 有对应 K/V，Cache 最大 |
| MQA | `H` | `1` | 所有 Query Head 共享 K/V，Cache 最小 |
| GQA | `H` | `G` | 每组 Query Head 共享 K/V，质量与成本折中 |

例如 Query Heads 为 32、KV Heads 为 8 时，每 4 个 Query Head 共享一组 K/V；相较 32 个 KV Head 的 MHA，KV Cache 的 Head 维约缩小为四分之一。

## Attention 在 Transformer Block 中的位置

Attention 不是整个 Transformer。常见 Pre-Norm Block 可概念化为：

```text
X₁ = X + Attention(Norm(X))
Y  = X₁ + MLP(Norm(X₁))
```

- `Norm` 稳定激活尺度；
- `Attention` 负责不同 Token 位置间的信息交换；
- 残差连接保留原信息并改善深层梯度传播；
- `MLP` 对每个位置独立进行非线性特征变换。

简化地说，Attention 主要负责“横向看其他 Token”，MLP 主要负责“纵向加工当前位置特征”。

## 训练与自回归推理的区别

### 训练：并行计算整个序列

```text
输入：我 喜欢 吃 苹果
目标：   喜欢 吃 苹果 EOS
```

已知完整文本，所以所有位置可在 Causal Mask 下并行预测各自下一个 Token。

### 推理：每次新增一个 Token

```text
我 喜欢 吃 → 预测 苹果 → 追加 → 继续预测
```

未来 Token 尚不存在，生成必须逐步进行。若每步重算全部历史 K/V，会产生大量重复工作，因此使用 KV Cache。

## KV Cache：为什么缓存 Key 和 Value

历史 Token 在某层生成的 K/V 后续通常不再变化，可以缓存：

```text
K_cache.shape ≈ [B,H_kv,S_history,d_head]
V_cache.shape ≈ [B,H_kv,S_history,d_head]
```

生成新 Token 时：

1. 只为新 Token 计算 `q_new、k_new、v_new`；
2. 把 `k_new、v_new` 追加到 Cache；
3. 用 `q_new` 与全部历史 Key 比较；
4. 对全部历史 Value 加权求和。

Query 表示当前新位置想查什么，下一步会产生新 Query；历史 K/V 会持续被未来 Query 读取，所以缓存 K/V。

KV Cache 容量近似正比于：

```text
Batch × 层数 × 序列长度 × KV Head 数 × Head 维度 × 2 × 元素字节数
```

详见 [推理与 KV Cache](/llm/inference)。

## 复杂度：二次增长发生在哪里

```text
Q.shape  = [S,d_k]
Kᵀ.shape = [d_k,S]
QKᵀ      = [S,S]
```

多头计算量通常写作 `O(S²D)`。朴素实现显式保存每头矩阵时，中间激活规模约为 `O(BHS²)`。

序列长度从 4K 增至 8K，分数矩阵元素约变为 4 倍。使用 KV Cache 的增量推理中，单个新 Token 只与 `S_history` 个 Key 比较，单步对历史长度近似线性；但生成完整序列的累计工作量仍呈二次级趋势。

## FlashAttention 优化了什么

FlashAttention 不把标准全局 Attention 改成近似算法，也不消除 `QKᵀ` 的二次算术量。它通过分块、在线 Softmax 和重计算，减少大型 Attention 矩阵在 GPU 高带宽内存与片上 SRAM 间的读写，并避免完整物化 `S×S` 中间矩阵。

因此它显著降低中间显存和内存 IO，通常也加速执行；标准密集 Attention 的算术复杂度仍近似 `O(S²D)`。

其他长上下文方案包括：

- Sliding Window Attention：只看附近固定窗口；
- Sparse Attention：只计算部分连接；
- Block/Chunk Attention：按块组织局部或分层交互；
- Linear Attention：改变标准 Softmax Attention 形式以避免显式 `S×S`。

选择方案还要考虑模型质量、Kernel 支持、硬件利用率、KV Cache 与目标上下文长度。

## PyTorch：从零实现多头因果 Self-Attention

下面代码强调形状与步骤，不含全部生产优化：

```python
import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,S,D]
        batch_size, seq_len, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # [B,S,D] → [B,H,S,d_head]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # [B,H,S,d] @ [B,H,d,S] → [B,H,S,S]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        # [B,H,S,S] @ [B,H,S,d] → [B,H,S,d]
        context = weights @ v

        # [B,H,S,d] → [B,S,D]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)
```

生产代码更常使用融合接口：

```python
from torch.nn import functional as F

output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)
```

框架可根据设备、类型和形状选择高效实现。手写步骤便于学习，但可能物化大型分数矩阵，也更容易出现 Mask、精度和性能问题。

## 常见实现错误与排查

1. **Softmax 维度错误**：应沿 Key 维，即最后一维归一化，每行权重和接近 1。
2. **Mask 方向反了**：Decoder 应允许主对角线和左侧，屏蔽右侧未来。
3. **Softmax 后才加 Mask**：Mask 应在 Softmax 前作用于 Logit。
4. **整行都被屏蔽**：全为 `-∞` 可能产生 NaN，应检查 Padding 与 Mask 合并。
5. **忘记除以 `√d_head`**：大维度下 Softmax 更易饱和。
6. **Head 维度错位**：持续检查 `[B,S,D] ↔ [B,H,S,d_head]`；transpose 后 view 常需 `contiguous()`。
7. **把权重当完整解释**：信息还经过 V、输出投影、残差、MLP 和多层组合。
8. **混淆 Mask 与 Cache**：Mask 防止看未来，KV Cache 避免重算历史，不能互相替代。

## 本章检查表

- 能区分 Query、Key 和 Value。
- 能从 `[B,S,D]` 推导多头 Q/K/V 形状。
- 能解释 `QKᵀ → scale → mask → softmax → ×V`。
- 能说明为什么缩放因子是 `√d_k`。
- 能辨认 Attention 矩阵的行是 Query、列是 Key。
- 能区分 Causal Mask 与 Padding Mask。
- 能说明多头结果怎样拼接并经 `W_O` 投影。
- 能区分 MHA、MQA、GQA 及其 KV Cache 代价。
- 能解释训练并行 Attention 与增量推理的区别。
- 能说明 FlashAttention 优化 IO 与中间显存，而非消除全部二次算术量。

### 检查表答案与原文依据

1. **Q、K、V 有什么区别？** Q 表示当前位置需要什么，K 用于匹配，V 是实际汇总的内容。见[直觉解释](#先建立直觉-attention-是一次-带条件的信息检索)和[Q、K、V 的产生](#q、k、v-是怎样产生的)。
2. **多头形状怎样变化？** `[B,S,D]` 拆分为 `[B,H,S,d_head]`。见[Multi-Head Attention](#multi-head-attention-为什么需要多个头)。
3. **标准顺序是什么？** 点积、缩放、Mask、按行 Softmax、对 V 加权。见[公式](#scaled-dot-product-attention-公式)和[完整手算](#一次完整手算-最后一个-token-如何读取前文)。
4. **为什么除以 `√d_k`？** 用于控制点积分数尺度，减轻 Softmax 饱和。见[缩放原因](#为什么要除以-√d-k)。
5. **矩阵行列是什么？** 行是 Query 目标位置，列是 Key/Value 来源。见[Attention 矩阵](#attention-矩阵的行列表示什么)。
6. **两类 Mask 有何区别？** Causal Mask 屏蔽未来，Padding Mask 屏蔽补齐。见[Causal Mask](#causal-mask-为什么生成模型不能看到未来)。
7. **多头怎样合并？** 每头独立计算，拼接后经 `W_O` 混合。见[Multi-Head Attention](#multi-head-attention-为什么需要多个头)。
8. **MHA、MQA、GQA 如何区分？** 主要区别是 KV Head 数分别为 `H、1、G`。见[MHA、MQA 与 GQA](#mha、mqa-与-gqa)。
9. **训练与推理有何区别？** 训练可并行处理全部位置，推理逐 Token 生成并缓存历史 K/V。见[训练与推理](#训练与自回归推理的区别)和[KV Cache](#kv-cache-为什么缓存-key-和-value)。
10. **FlashAttention 改变什么？** 分块并使用在线 Softmax，降低 IO 和中间显存，不把密集 Attention 变为线性算术复杂度。见[FlashAttention](#flashattention-优化了什么)。

## 中文延伸学习资源

- [Transformer 论文逐段精读](https://www.youtube.com/watch?v=nzqlFIcCSWQ) — 作者/频道：跟李沐学 AI；平台：YouTube。以原始论文为主线讲解 Scaled Dot-Product Attention、Q/K/V、Multi-Head Attention、Mask 与复杂度。
- [Transformer 架构](https://infrasys-ai.github.io/aiinfra-docs/06AlgoData01Basic/README.html) — 来源：AIInfra 中文开源课程。除 MHA 外还覆盖 MQA、GQA、MLA，适合继续研究现代大模型如何降低 Attention 与 KV Cache 开销。
- [Happy-LLM](https://github.com/datawhalechina/happy-llm) — 来源：Datawhale 开源社区；平台：GitHub。第 2 章包含 Transformer/Attention 原理和代码实现，第 5 章进一步使用 PyTorch 搭建并训练小型 LLM。

> **版权与来源说明：** 本节只提供外部资源索引和原创导读，不复制视频、课程正文、图示或源代码。资源版权归对应作者、项目及平台所有；使用代码时请检查仓库 LICENSE。链接失效或内容更新时，以原始发布页为准。

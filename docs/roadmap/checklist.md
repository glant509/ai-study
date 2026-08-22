# 学习检查表

## LLM

- [ ] 能解释 token、tensor shape、causal attention 与 RoPE。
- [ ] 能区分 prefill、decode、TTFT、TPOT。
- [ ] 能估算 KV Cache 随并发和上下文增长的代价。

## Agent

- [ ] 不用框架写出受预算约束的 Agent loop。
- [ ] 工具调用具备 schema、超时、幂等与错误分类。
- [ ] 长任务可 checkpoint、暂停、审批和恢复。

## Knowledge & Protocol

- [ ] RAG 有检索与生成分层评测集。
- [ ] 能实现并安全接入一个 MCP Server。

## Production

- [ ] Trace 能串联模型、检索、工具与状态变化。
- [ ] 敏感操作有最小权限、沙箱和审批。
- [ ] 用真实流量分布完成负载测试与容量规划。

全部勾选后，回到作品项目做一次故障演练：模型超时、工具重复执行、索引撤权和沙箱逃逸尝试分别如何被发现与处理？

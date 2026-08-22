# Runtime 循环

一个最小 Runtime 只需状态、模型和工具三类接口。

```python
for step in range(max_steps):
    messages = context_builder.build(state)
    decision = model.generate(messages, tools=registry.schemas())
    if decision.is_final:
        return decision.answer
    result = registry.execute(decision.tool_call)
    state.append(decision, result)
raise StepLimitExceeded()
```

## 生产边界

- 每个 tool call 使用唯一幂等键。
- 重试区分瞬时错误、业务拒绝与模型错误。
- 超时和取消信号贯穿模型、工具与存储层。
- 所有状态转换写入事件日志，便于恢复和审计。

Runtime 应掌控执行权。模型提出意图，系统验证并执行；不要让模型输出直接成为系统命令。

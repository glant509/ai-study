# Checkpoint 与恢复

长任务可能因进程重启、人工暂停、工具超时或预算耗尽中断。Checkpoint 让执行从明确状态恢复，而不是重放整个对话猜测进度。

```text
checkpoint = {
  run_id,
  version,
  state,
  next_action,
  completed_effects,
  pending_approval,
  context_refs
}
```

## 一致性

外部副作用和 checkpoint 之间存在经典双写问题。使用幂等键、outbox、状态机或可补偿操作，确保恢复后不会重复付款、发信或创建资源。

Checkpoint schema 要版本化；升级 Runtime 时提供迁移逻辑，无法迁移则安全终止并请求人工处理。

# Tool Calling

工具把模型的自然语言决策连接到确定性系统。一个好工具应语义单一、schema 明确、错误可理解、结果紧凑。

```json
{
  "name": "get_order",
  "description": "按订单号查询订单状态",
  "parameters": {
    "type": "object",
    "properties": { "order_id": { "type": "string" } },
    "required": ["order_id"],
    "additionalProperties": false
  }
}
```

## 设计原则

1. 用业务动作命名，不暴露底层表结构。
2. 读取与写入工具分开，写操作要求更高权限。
3. 返回稳定的结构化错误码，同时给模型短解释。
4. 对转账、删除、发布等高风险动作增加人工确认。

工具结果不是可信文本。网页、邮件、文档都可能包含提示注入，必须按数据处理。

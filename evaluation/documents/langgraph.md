# LangGraph 工作流

LangGraph 使用 State、Node 和 Edge 表达有状态 Agent 工作流。State 保存节点之间需要共享的数据；Node 执行检索、生成、验证或工具调用；Edge 决定下一个节点，条件 Edge 可以根据意图或验证结果进行路由。

图状态必须可以序列化，因此不能把数据库连接、SDK 客户端、向量索引实例或 API Key 放进 State。外部资源应由应用依赖注入，State 只保存用户 ID、线程 ID、问题、检索片段和验证错误等数据。

Checkpoint 保存图在特定线程中的执行状态，使中断任务能够恢复并支持多轮对话。Checkpoint 不等同于长期记忆。反思和修复边必须设置明确的最大次数，例如只允许一次修复，防止 Agent 陷入无限循环并持续消耗模型额度。

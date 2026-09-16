# TokenPilot

Python 3.9+，无运行时第三方依赖。当前包含资源账本、实验比较、Baseline Runner 与 OpenAI-compatible adapter。**验证仍全部使用人工构造的离线样本，没有真实模型 benchmark 结果。**

## 运行

在项目根目录执行，无需设置 PYTHONPATH，也无需先安装：

```sh
python3 -m unittest -v
python3 -m benchmarks.smoke_baseline
python3 -m benchmarks.smoke_provider
```

- `smoke_baseline` 验证模拟成本比较。
- `smoke_provider` 验证 adapter → usage → ledger → evaluator，只回放包内 JSON，不读取密钥、不联网。
- 输出中的 `SIMULATED_SMOKE_TEST_ONLY` 必须随结果保留。样本里的 Token、费用、质量分数不是实测数据。

不要使用 `python3 benchmarks/smoke_baseline.py` 作为源码运行入口。
可选：在自己的虚拟环境里执行 `python3 -m pip install -e .`，安装后可使用 `tokenpilot-smoke`。构建依赖可能需要下载；本轮开发未安装依赖。

## 最小结构

| 模块 | 职责 |
| --- | --- |
| `tokenpilot/telemetry` | Ledger 资源记录、ExperimentRun 与墙钟计时 |
| `tokenpilot/benchmark/compare.py` | 含 overhead 的总成本比较与质量门槛 |
| `tokenpilot/benchmark/runner.py` | 单任务、单次调用，先记账再评估 |
| `tokenpilot/providers/base.py` | 不绑定供应商的请求、响应、Usage、ProviderError 接口 |
| `tokenpilot/providers/openai_compatible.py` | 非流式 Chat Completions 文本请求转换及响应解析 |
| `tokenpilot/providers/transport.py` | 可注入传输接口；默认禁止联网的标准库 HTTPS 实现 |
| `tokenpilot/providers/normalization.py` | 严格校验 Chat Completions usage |
| `benchmarks/fixtures` | 人工构造的公开测试数据，不含用户请求或凭据 |

## Adapter 合同

初始化时显式指定供应商名称和 API 根地址，例如 `https://fixture.invalid/v1`；adapter 追加 `/chat/completions`。不推测供应商、不选择模型、不读取环境变量、`.env` 或钥匙串。

- 一次请求只有一条 user 文本消息、一个 completion，`stream=False`、`n=1`。
- 默认将 `max_output_tokens` 转换为 `max_completion_tokens`；旧兼容服务可显式选 `token_limit_field="max_tokens"`，不会失败后重发。
- `temperature=None` 表示省略该字段；未提供 seed 时同样省略。不同模型支持的参数由调用方确认。
- usage 必须包含非负整数 input/output；缓存是 input 子集；若提供 total，必须等于 input + output。reasoning_tokens 等细分值不再次相加。
- 截断、拒绝、工具调用、异常结束原因不进入质量评估。有效 usage 与费用仍尽量保留。
- `ProviderError` 只包含固定错误代码、HTTP 状态和可选部分响应。Runner 不存原始错误正文、请求头或提示词。
- 标准库传输默认禁用；启用后也无自动重试、无重定向、无环境代理发现，校验 HTTPS 并限制响应大小。timeout 是底层连接/读操作超时，不是绝对总时限。
- 仅支持这个文本子集，不覆盖所有 OpenAI-compatible 服务的扩展、多模态、Responses API 或工具执行。

官方字段参考：[OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)。接口可兼容不代表每个模型都支持相同参数。

## 成本来源

标准 token usage 不是账单金额。默认 `cost_usd=None`。需要一个显式配置并带版本号的 `billing_extractor`，从所选供应商有文档依据的计费响应中返回 `CostEvidence`；这一步尚未对接具体供应商账单。

| 类型 | 处理方式 |
| --- | --- |
| `reported` | 有来源的已报告美元金额；真实运行可计入总成本 |
| `estimated` | 仅保留在事件的 `unverified_cost_usd`，不冒充实付成本，账目不完整 |
| `simulated` | 仅允许离线模拟成本；真实运行不能据此完成记账 |
| 未知 | 记录已知 Token，成本为占位零且 `cost_known=False`；运行失败，禁止更优结论 |

离线传输的证据统一标记 `simulated`。示例的 `_fixture_cost_usd` 是**自定义测试字段**，不是 OpenAI 的标准响应字段。`CostEvidence.source`、供应商名称和版本标识应只含公开说明，不能放密钥、账单个人信息或原始 HTTP 头。

## 实验约束

1. total tokens = input + output；cached input 不重复相加。
2. 净节省 = baseline 总成本 − candidate 总成本。candidate 总成本已含 overhead；优化调用必须另记 `is_overhead=True` 事件，不能再扣一次。
3. 更优结论要求双方成功、账目完整和明确通过质量门槛。质量值必须有限；缺少门槛不能推断质量合格。
4. 只比较同一 task、相同数据集/评估器版本、相同模拟属性的运行。模拟比较的序列化结果本身也带警告。手工构造的旧 run 若双方都没有版本信息，只能按旧约定比较，调用方负责完善来源。
5. 保存请求哈希、模型、生成参数、数据集/评估器版本、adapter 版本、端点哈希、字段选择、超时与计费解析器版本；响应保留有限的模型/ID/系统指纹。调用方需保留版本化输入与评估器源码；哈希不能恢复输入，也不保证远程模型确定性。
6. Provider 异常可能仍产生费用；缺失响应时账目不完整，需向供应商对账。有效响应后的解析或评估失败保留已知费用。即使用量缺失，独立有效的费用也会入账，同时标记 `usage_known=False`、Token 为占位零、账目不完整。
7. Runner 要求显式 `allow_paid_api=True` 才能进入真实 Provider；标准库传输还要求 `allow_network=True`，并显式传入凭据。这些开关不是用户授权，也不是预算控制器。
8. 评估器目前必须是本地、无额外付费调用的函数。付费评估器和候选优化的开销接入尚未实现，不能偷偷放进 callback。
9. Ledger 的公开事件列表仍可由调用方修改；append-only 是使用约定，不是防篡改存储。异常记录不包含完整堆栈或 HTTP 原文。

## 下一步

用户确认供应商、具体模型和首次实验预算后，再确认参数支持、计费字段及计费解析器，完成对应脱敏样本测试，并设计可记录失败费用的最小真实实验。已提供显式 opt-in 的真实 API smoke 入口；候选优化、批处理、对账持久化和预算控制仍待后续实现。

## 真实 Moonshot usage smoke（显式付费调用）

默认 `python3 -m benchmarks.smoke_live_provider` 不读取 `.env`、不联网。
显式调用前在项目根目录本地 `.env` 配置 `MOONSHOT_API_KEY`、`MOONSHOT_BASE_URL`，
可选 `MOONSHOT_MODEL=kimi-k2.6`。环境变量优先于 `.env`，`--model` 优先于两者。
`.env` 只解析这些键的简单 `KEY=value`（支持配对引号和 export 前缀），不执行 shell、不做变量展开。
不要提交 `.env` 或粘贴密钥。

```sh
cd ~/Desktop/tokenpilot
python3 -m benchmarks.smoke_live_provider --allow-paid-api --model kimi-k2.6 --max-output-tokens 1024
```

只发一次请求，不重试；不额外设置 thinking，保留模型默认行为，省略 temperature/seed。
1024 是包含 reasoning 的输出 token 上限，不是金额预算。返回 length 时保留 usage，smoke 退出码为 1。
收到完整 `TOKENPILOT_OK` 时 `smoke_ok=true`、退出码为 0；这只表示连通性及回复验证通过。
没有 billing evidence 时 `experiment_status=failed`、`accounting_complete=false` 是现有记账规则，
不会更改实验统计或把未知费用当成免费。配置错误退出码为 2。输出不含原始回复或密钥，不自动保存文件。

usage 支持 prompt/input、completion/output 别名，缓存支持嵌套 cached_tokens 及顶层 cached_tokens；
同义字段同时出现必须一致，不能相加。reasoning_tokens 是可选 output 明细，缺失为 None，
只写入 Ledger 事件 `metadata.usage_breakdown`，要求为非负整数且不超过 output。
所有 total 始终等于 input + output。未知扩展字段不保存，避免保留原始响应敏感信息。
字段依据：[Moonshot 官方 Kosong 实现](https://moonshotai.github.io/kosong/kosong/chat_provider/kimi.html)、
[OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat)。
`tests/fixtures` 中的三个样本均已去掉真实标识及推理正文；缓存/成功样本为合成数据，
截断样本复现已知的 16 input + 20 output = 36 total，其中 19 reasoning 不再次计入。

## v0.2.0：结构化 benchmark（开发中）

已加入 12 个固定任务、full-history / sliding-window / retrieval-context /
TokenPilot 四种策略、确定性答案与引用校验、逐条落盘和显式预算预检。
这是软件里程碑，**没有新增真实模型节省证据**。完整运行说明、计费范围及限制见
[Benchmark guide](docs/BENCHMARK.md)。上方 v0.1 说明描述旧 smoke/ledger 接口；
新 batch runner 单独区分 measured / estimated / simulated / unknown。

```sh
python3 -m benchmarks.run --dry-run
python3 -m benchmarks.run
```

默认离线；未指定本地 CPU 价格时净成本账目保持不完整。模拟 token 是字节单位，
不能当成模型实测 token。当前还没有选择开源许可证，也未发布 v1.0。

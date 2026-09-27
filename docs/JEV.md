# Optional Jev research review

[Home](../README.md) · [中文入口](../README.zh-CN.md) · [Research methods](METHODS.md) · [Historical benchmark](BENCHMARK.md)

AlphaResearchOS can add TypeSafe's Jev as a second, structured review of **development evidence**. It runs after the existing independent model reviewer approves a candidate, before that candidate becomes eligible for selection. It is disabled by default and applies to agent research mode.

Jev's System One API accepts a state and typed questions; it is used here for judgment over supplied evidence, not feature generation, model fitting, return prediction, or order execution. The integration uses the [official HTTP contract](https://docs.typesafe.ai/api) and defaults to the fixed version `jev-1.13.0`. This release has been validated with offline, mocked responses; a live Jev experiment and a with/without-Jev performance comparison have **not** been run. The [sector ETF results](BENCHMARK.md) predate this integration.

## How the gate works

```text
Proposal → causal checks + development evaluation → independent model review
                                                    ↓ approve
                                            optional Jev review
                                                    ↓ approve
                                         development-score selection → holdout
```

One HTTP request contains a three-way Choice (`approve`, `revise`, `reject`) and four Noul checks:

| Check | What the supplied evidence must support |
| --- | --- |
| `hypothesis_alignment` | Features and model implement the stated research hypothesis |
| `cost_support` | Cost and turnover claims agree with measured development results and execution settings |
| `fold_consistency` | Stability claims acknowledge weak or conflicting development folds |
| `evidence_sufficiency` | The candidate and independent review have enough evidence for their claims |

The local code controls eligibility. Approval requires the returned Choice to be `approve`, its confidence and `approve` probability to meet the configured threshold, and **all four check values** to meet the same threshold. A reliable `reject` stays rejected; uncertain answers and failed checks become `revise`. Connection failures, malformed responses, missing budgets and timeouts do not silently approve a candidate. A probability here describes a review judgment, not the probability of earning a return.

The request includes allowlisted candidate fields, the independent review, development metrics/folds/baselines, execution assumptions and technical checks. Holdout results are excluded from this review context. Existing causal checks, constraints and final holdout isolation still apply.

## Configure in the workbench

1. Open **Settings / 设置** and keep the normal Codex or compatible API connection configured.
2. Enable **Jev**, use the defaults below, and enter your TypeSafe API key.
3. Save. Optionally select **Test Jev connection / 测试 Jev 连接** to send one real provider request. Saving settings alone makes no request.
4. Start a new agent research experiment. An approved candidate's trajectory shows the Jev decision, checks, confidence, probabilities and usage metadata.

Enabling the gate sends the development context to the configured TypeSafe-compatible endpoint. The API key stays in the local settings store and is omitted from public settings responses and research reports. Remote provider calls, including the explicit connection test, may be billed by that provider; the connection test is separate from an experiment's budget.

| Settings field | Default | Meaning |
| --- | --- | --- |
| `jev_enabled` | `false` | Enable the extra review in agent mode |
| `jev_base_url` | `https://api.typesafe.ai/v1` | Provider root or `/v1` base; the client calls `/v1/systemone` |
| `jev_model` | `jev-1.13.0` | Fixed version; `jev-latest` and `jev-preview` aliases are also accepted |
| `jev_api_key` | empty | Separate TypeSafe key |
| `jev_min_confidence` | `0.7` | Shared approval threshold, greater than `0.5` and at most `1` |

These are flat fields in `POST /api/settings`, alongside the normal provider fields. Leaving a key blank retains the saved value; `clear_jev_api_key: true` explicitly clears it. Prefer the settings form for entering secrets.

The same values can be supplied before starting the application through `ALPHAOS_JEV_ENABLED`, `ALPHAOS_JEV_BASE_URL`, `ALPHAOS_JEV_MODEL`, `ALPHAOS_JEV_API_KEY` and `ALPHAOS_JEV_MIN_CONFIDENCE`. `TYPESAFE_API_KEY` is the key fallback when `ALPHAOS_JEV_API_KEY` is unset. See [`.env.example`](../.env.example); the application does not automatically load arbitrary `.env` files. Explicitly saved settings take precedence over environment values.

## Budgets and audit

Each attempted Jev review consumes one request from the **existing model-request budget** and reserves **32,768 tokens** from the existing token-admission budget before the call. This is a conservative local admission allowance, not a price estimate or a claim about actual usage. The provider's reported token usage is recorded separately. Failed requests still consume their reserved budget. A candidate with proposal, independent review and enabled Jev approval therefore needs up to three requests; raise the campaign budget deliberately when comparing equal candidate counts.

Requests have no automatic retries or redirects, a 24 KB request limit, a 32 KB response limit, and a timeout bounded by the remaining campaign time and 90 seconds. Decisions are stored as `jev_gate`; provider, returned model version, usage and duration are stored as `jev_usage`. `jev_calls` tracks gate attempts within the campaign's total model calls. Fixed model IDs must match the provider response; a model alias changing versions during the experiment prevents mixing its judgments. Changing Jev settings on a paused experiment requires restoring its original settings or creating a new experiment.

This feature adds no broker connection, order execution, autonomous live trading, or historical benchmark rerun. For a future efficacy study, keep the dataset, proposal/review providers and budgets fixed, compare Jev on/off in separate campaigns, record every attempted candidate, and evaluate only after freezing selection. No such uplift is claimed here.

## Design references

| Project inspected on 2026-09-28 | Relevant idea | Scope of reuse |
| --- | --- | --- |
| [QuantDinger, commit `709521ffd6`](https://github.com/OpenByteInc/QuantDinger/blob/709521ffd6df8460dc7bfaba0451258b0f1743da/backend_api_python/app/services/ai_decision_filter.py) · [Apache-2.0](https://github.com/OpenByteInc/QuantDinger/blob/709521ffd6df8460dc7bfaba0451258b0f1743da/LICENSE) | Atomic checks, a code-controlled final decision and auditable structured results | Design reference; no source or prompts copied. Its live-entry/fail-open policy is not adopted |
| [Jev-Trades, commit `25f015ca`](https://github.com/zadescoxp/Jev-Trades/blob/25f015cae30a07fae2dfab570d0519e559d50c78/pipeline/schema.py) · [Apache-2.0](https://github.com/zadescoxp/Jev-Trades/blob/25f015cae30a07fae2dfab570d0519e559d50c78/LICENSE) | Typed judgments, confidence display and decision history | Design reference; no trading, broker or portfolio implementation copied |

The AlphaResearchOS client and research questions are independently implemented against TypeSafe's API. No upstream application is installed as a dependency, and no model weights are included. The application licenses above do not grant or cover TypeSafe API access.

## 中文使用说明

Jev 是**默认关闭的附加开发期复核**：只有原独立模型审核通过后才调用，全部检查和决策门槛满足后候选才能参与选择。四项检查分别关注假设与实现、成本依据、开发折一致性和证据充分性；不向它提供留出结果，也不接入实盘交易。

在设置中启用 Jev，填写独立的 TypeSafe 密钥并保存。默认地址为 `https://api.typesafe.ai/v1`，固定模型为 `jev-1.13.0`，阈值为 `0.7`。保存本身不调用服务；点击连接测试会发送一次可能计费的请求。正式研究中每次 Jev 尝试计入原请求预算，并预留 32,768 token 准入额度，实际用量另行记录。失败、低置信度或证据不足不会自动放行。开启后，同样候选数通常需要更多请求预算。

当前验证使用离线夹具和模拟响应，未执行真实 Jev 调用或收益消融。README 的行业 ETF 业绩全部来自接入 Jev 之前的历史实验，不能解释为 Jev 带来的收益。

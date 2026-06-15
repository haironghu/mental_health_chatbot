# 多 Agent 架构设计

本文档描述从当前「单 LLM 调用」版本演进到多 Agent 架构的设计与路线图。

## 设计原则

**FSM 仍是确定性的「大脑」，Agents 是它的「专家」。**

- Agents 之间**不互相决策**，各自只负责擅长的事，由 Coordinator（协调器）统一调度
- 心理健康产品必须**可预测**，避免多 Agent 自由对话导致失控
- 危机干预流程**不可被 LLM 自由意志绕过**
- 出问题时能快速定位是哪个 Agent 出错

## Agent 划分

| Agent | 职责 | 触发时机 | 模型档位 |
|---|---|---|---|
| **Triage 分诊** | 提取 R(t) 信号（s_emotion / s_keyword / s_behavior）+ 语言检测 | 每轮 | 便宜（Haiku / Flash Lite） |
| **Safety Monitor 安全监控** | 独立检测自伤/自杀/危机意念 | 每轮（并行） | 便宜 + 特化 |
| **Boundary 边界守护** | 偏题 / 药物咨询 / 技术支持等分类 | 每轮（并行） | 便宜 |
| **K6 Scorer 评估** | 根据对话推断 K6 六维度分数 | 仅 K6_ASSESSMENT 状态 | 中（Haiku / Sonnet） |
| **Strategy Selector 策略选择** | K6 完成后选 PM+ 策略 + 给出可审计的推荐理由 | 一次性，K6 完成时 | 中 |
| **Therapist 治疗师** | 自然语言回复 | 每轮 | 好（Sonnet / Opus） |
| **Memory 记忆** | 摘要长对话历史，减少上下文 | 每 N 轮一次 | 便宜 |

## 处理流程（并行 + 串行）

```
用户消息
  ↓
┌──────────────────────────────────┐
│  并行调用（< 1s）：               │
│   - Triage                        │
│   - Safety Monitor                │
│   - Boundary                      │
│   - K6 Scorer（仅 K6 阶段）       │
└──────────────────────────────────┘
  ↓
确定性逻辑（Coordinator）：
  - 更新 R(t)、K6 分数
  - FSM 决定下一状态
  - K6 完成 → 调用 Strategy Selector
  ↓
串行调用：
  - Therapist（生成回复）
  ↓
（可选）回复后安全复审
  ↓
发送给用户
```

并行的分诊/打分任务跑得快又便宜，真正花钱的只有 Therapist。

## 通信模式：Hub-and-spoke

Agents 之间不直接通信，全部通过 Coordinator 中转。

```python
class Coordinator:
    def process(self, user_msg):
        signals = parallel_call([triage, safety, boundary, k6_scorer])
        self.fsm.update(signals)          # 确定性
        if signals.crisis_detected:
            self.fsm.force_crisis()
        response = therapist.generate(self.fsm.state, signals)
        return response
```

理由：可预测、易测试、好排查。不采用 agent-to-agent（如 CrewAI 风格）的自由对话——不应让 Triage Agent「决定」是否调用 Crisis Counselor。

## 成本与延迟优化

| 优化 | 收益 |
|---|---|
| Triage / Safety / Boundary 用 Haiku / Flash Lite | 成本几乎可忽略 |
| Therapist 按严重度切换档位（mild 用 Haiku，severe 用 Sonnet） | 30-50% 总成本 |
| Memory Agent 每 5 轮跑一次并缓存摘要 | 大幅减少 Therapist 上下文 |
| 危机时跳过其他 Agent，直接调 Crisis Therapist | 危机响应延迟 < 1s |

预期每轮 3-4 次 LLM 调用（当前 2 次），总成本约 1.5-2x，质量明显提升。

## 框架选择

| 选项 | 评价 |
|---|---|
| **自研**（在现有 orchestrator 上扩展） | 推荐，FSM / Prompt / Session 都现成 |
| Claude Agent SDK | 可选，官方支持，原生工具调用 |
| LangGraph | 适合复杂图状流转，对当前 FSM 偏重 |
| CrewAI | 不推荐，鼓励自由对话，违背确定性原则 |

## 分阶段路线图

不要一次推翻，建议按以下顺序，每阶段独立可发布：

```
现状：Analysis（混合）+ Response（混合）
        ↓
Phase 1 ✅：拆出 Triage + K6 Scorer（把 Analysis 拆开），建立 Agent 框架 + Coordinator
        ↓
Phase 2 ✅：加 Safety Monitor（并行运行）+ 确定性关键词兜底 + 三重危机判定
        ↓
Phase 3（跳过）：原计划 Therapist 拆成多专家，评估后认为过度设计。
        现有「单 Therapist + 按状态切换 task 模板」已实现专家行为；
        危机已改为固定消息（不经 LLM），故无需再拆。
        ↓
Phase 4 ✅：加 Memory Agent（长会话摘要，降低 token 成本）
        ↓
Phase 5：加 Strategy Selector，记录每次策略决策理由（可审计）
```

### Phase 1 已实现（当前代码）

```
app/agents/
├── base.py              # Agent / AnalysisAgent / ResponseAgent 基类 + AgentContext
├── triage.py           # TriageAgent（每轮，R(t)信号+危机+语言+意愿，便宜模型）
├── k6_scorer_agent.py  # K6ScorerAgent（仅 K6 阶段，六维度评分，中档模型）
├── therapist.py        # TherapistAgent（每轮，自然语言回复，质量优先）
└── coordinator.py      # Coordinator（并行调度 + 确定性决策 + 回复）
```

- 分析 Agent 通过 `ThreadPoolExecutor` 并行运行
- K6ScorerAgent 仅在 K6_ASSESSMENT 状态运行，省成本
- 每个 Agent 失败时返回安全默认值，不拖垮流水线
- Coordinator 输出 `trace`（各 Agent 耗时），写入日志供观测
- 各 Agent 模型档位可在 `.env` 配置（MODEL_TRIAGE / MODEL_SAFETY / MODEL_K6 / MODEL_THERAPIST）

### Phase 2 已实现（Safety Monitor）

```
app/agents/safety_monitor.py    # 专注危机/自伤/自杀意念检测（每轮并行）
app/safety/crisis_keywords.py   # 确定性危机关键词兜底（粤/普/英）
```

- **职责分离**：危机检测从 TriageAgent 分离到独立的 SafetyMonitorAgent，
  专注的 prompt 比混在分诊里更可靠；s_keyword 也随之移到 safety
- **三重危机判定**（任一命中即强制危机干预，冗余保障安全）：
  1. SafetyMonitorAgent 的 LLM 判断（crisis_detected）
  2. 确定性关键词兜底（contains_crisis_keywords）—— LLM 失败也能 catch 明显危机
  3. R(t) 红色预警
- 关键词命中会写入 trace（crisis_keyword_hit）供审计

### Phase 4 已实现（Memory Agent）

```
app/agents/memory.py            # 滚动摘要：压缩较早对话
app/prompts/agents/memory.jinja2
```

- **目的**：长会话每轮把全部历史发给 4 个 agent，token 成本随轮数线性增长。
  Memory Agent 把较早对话压成摘要，每轮只发「摘要 + 最近 4 轮」。
- **窗口策略**（在 Coordinator._prepare_memory）：
  - 短会话（≤ 8 轮）：发送全部历史，无摘要
  - 长会话：最近 4 轮原始 + 较早内容摘要
- **更新节奏**：每 `memory_summary_every`（默认 5）轮更新一次摘要，或首次超阈值时立即生成；其余轮复用已有摘要（不调 LLM）
- 摘要通过 `_prepend_summary` 注入每个 agent 的 system prompt 顶部
- 摘要更新耗时写入 trace（memory_ms）
- 配置：MODEL_MEMORY / MEMORY_SUMMARY_EVERY / RECENT_TURNS_WITH_SUMMARY / RECENT_TURNS_NO_SUMMARY

## 可观测性（必做）

多 Agent 最大的坑是「哪里出错了不知道」。需要：

1. **Trace ID**：每条用户消息生成一个 trace_id，所有 Agent 调用挂在该 trace 下
2. **每个 Agent 的日志**：输入 / 输出 / 延迟 / 成本
3. **决策日志**：FSM 状态变化、策略选择、危机触发，全部记录原因
4. **简单 dashboard**：按用户 hash 查看完整对话流（Agent 调用图）

不做这一层，多 Agent 几乎无法调试。

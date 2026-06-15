# 心理健康 WhatsApp 聊天机器人（多 Agent 版本）

面向香港青少年的匿名心理健康支援聊天机器人。通过 WhatsApp 接入，使用 LLM 辅助 **K6 凯斯勒心理困扰量表**评估用户心理状态，并基于 **PM+（Problem Management Plus，WHO 设计）** 准则提供聊天安抚与干预；在识别危机信号时主动引导专业资源。

后端采用**多 Agent 架构**：一个确定性的协调器（Coordinator）调度多个专职 Agent（分诊 / 安全监测 / K6 评分 / 记忆 / 治疗师），状态流转由有限状态机（FSM）控制，LLM 只负责自然语言与信号提取，不掌控流程。设计原则与演进路线见 [docs/MULTI_AGENT_DESIGN.md](docs/MULTI_AGENT_DESIGN.md)。

> ⚠️ **本机器人是情绪支援与筛查工具，不能替代专业心理治疗或诊断。** 上线前 `app/safety/crisis_response.py` 中的固定危机消息须经专业 / 法务审核。

---

## 核心特性

- **K6 心理困扰评估**：LLM 在自然对话中推断 K6 六维度分数（紧张 / 无助 / 焦躁 / 抑郁 / 费力 / 无价值），跨轮取最大值平滑累积，输出 mild / moderate / severe 严重度
- **PM+ 干预策略**：根据 K6 结果选择对应策略 —— 管理压力（呼吸训练）、解决问题、行为激活、强化社交支持
- **实时风险监控 R(t)**：加权累积风险评分 + 四级预警（green / yellow / orange / red），独立于 K6，负责当下危机检测
- **三重危机判定**：安全监测 Agent + 确定性关键词兜底 + R(t) 红色，任一命中即强制危机干预（冗余保障）
- **危机固定回复 + 会话锁定**：检测到危机后不再用 LLM 即兴回复，只发预审固定消息，避免担责
- **规则驱动 FSM**：状态流转完全由规则决定，LLM 只负责生成，保证危机流程不被绕过
- **类人对话体验**：消息防抖（合并连发的短消息）、回复分段、打字延迟模拟、回复长度自适应
- **长会话降本**：Memory Agent 把较早对话压成摘要，每轮只发「摘要 + 最近几轮」
- **可观测 + 可审计**：每轮输出各 Agent 耗时 trace；PM+ 策略选择记录决策依据到审计日志
- **隐私设计**：手机号 SHA-256 哈希存储，本地 JSON 存储，无身份资料收集

---

## 技术栈

| 组件 | 说明 |
|---|---|
| Python 3.14 | — |
| [neonize](https://github.com/krypton-byte/neonize) | whatsmeow 的 Python 绑定，直连 WhatsApp Web 多设备协议（无需 Twilio / webhook） |
| OpenRouter | OpenAI 兼容 API 网关，可按 Agent 配置不同模型档位 |
| Jinja2 | 多层 Prompt 模板 |
| pytest | 161 项单元测试 |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填入 OPENROUTER_API_KEY

# 3. 启动（首次运行终端会显示 QR 码）
python main.py
```

登录：手机 WhatsApp → **设置 → 已连接的设备 → 连接设备** → 扫描终端中的 QR 码。
看到 `WhatsApp 已连接，等待消息...` 即成功，之后让用户给该账号发消息即可。

会话密钥保存在 `whatsapp_session.sqlite3`，后续启动自动重连，无需重复扫码（除非该文件被删除、在手机端登出、或超过 14 天未用）。

> **Windows 提示**：在 PowerShell / CMD 中运行 `python main.py`（QR 码已适配 Windows 终端编码）。WSL 需单独安装依赖。

---

## 对话流程

```
WELCOME（欢迎）
   ↓
K6_ASSESSMENT（自然对话中评估 K6 六维度）
   ↓ K6 完成（≥4 维度有信号 + 聊够 5 轮）
PM+ 策略（按 K6 严重度与最高维度选择）
   ↓ 聊够 5 轮
PM_DECISION（询问用户是否继续）
   ↓ 想继续 → 下一策略    想结束 → CLOSURE
CLOSURE（个性化告别）

CRISIS_INTERVENTION（任何时刻三重危机判定命中即强制进入；进入后会话锁定）
```

用户随时发送「開始」可重置会话重新开始（包括从危机锁定中逃出）。

---

## 多 Agent 架构

```
orchestrator.process()              # 会话生命周期（加载 / 重置 / 终止 / 危机锁定 / 持久化）
   └─> Coordinator.run()            # hub-and-spoke 调度（确定性大脑）
         ├─ Memory：派生历史窗口（长会话用「摘要 + 最近 4 轮」）
         ├─ 并行：TriageAgent + SafetyMonitorAgent + K6ScorerAgent（仅 K6 阶段）
         ├─ 三重危机判定：Safety LLM / 确定性关键词兜底 / R(t) 红色
         │     命中 → 强制危机 + 返回固定消息（跳过 LLM 回复）
         ├─ 确定性：R(t) 更新 / K6 更新 / FSM 决策 / 决策审计日志
         └─ TherapistAgent（生成回复）
```

| Agent | 职责 | 运行时机 | 模型档位 |
|---|---|---|---|
| TriageAgent | 情绪 / 行为信号 + 语言 + 用户意愿 | 每轮 | 便宜 |
| SafetyMonitorAgent | 危机 / 自伤 / 自杀意念检测 | 每轮 | 便宜 |
| K6ScorerAgent | K6 六维度评分 | 仅 K6_ASSESSMENT | 中档 |
| MemoryAgent | 较早对话摘要 | 每 N 轮 | 便宜 |
| TherapistAgent | 自然语言回复 | 每轮（危机时跳过） | 质量优先 |

### 危机处理策略（安全决策）

检测到危机时：

1. **不调用 LLM 生成回复** —— 改发预先写好、经审核的**固定危机消息**（热线 + 联系真人 + 999）
2. **会话锁定** —— 进入危机后只重复固定消息，不自动恢复正常聊天（即使用户情绪平复）
3. **可重置逃出** —— 用户发送「開始」可重置会话

固定危机消息见 `app/safety/crisis_response.py`，**须经专业 / 法务审核后定稿**。

---

## 项目结构

```
app/
├── config.py                  # 配置（pydantic-settings）
├── agents/                    # 多 Agent 层
│   ├── base.py                # Agent 基类 + AgentContext
│   ├── triage.py              # 分诊 Agent
│   ├── safety_monitor.py      # 安全监测 Agent
│   ├── k6_scorer_agent.py     # K6 评分 Agent
│   ├── memory.py              # 记忆 Agent（长会话摘要）
│   ├── therapist.py           # 治疗师 Agent
│   └── coordinator.py         # 协调器（并行调度 + 确定性决策 + 审计日志）
├── orchestrator/
│   ├── fsm.py                 # 有限状态机（WELCOME→K6→PM+→CLOSURE / CRISIS）
│   └── orchestrator.py        # 会话生命周期（委托 Coordinator）
├── intelligence/
│   ├── llm.py                 # OpenRouter 封装（支持按 agent 选模型）
│   └── prompt_builder.py      # 多层 Prompt 组装 + 摘要注入
├── safety/
│   ├── k6_scorer.py           # K6 评分器 + PM+ 策略选择（含依据）
│   ├── risk_monitor.py        # R(t) 风险公式 + 四级预警
│   ├── crisis_keywords.py     # 确定性危机关键词兜底
│   └── crisis_response.py     # 固定危机消息（须审核）
├── storage/
│   └── session_store.py       # 本地 JSON 会话存储（手机号哈希）
├── whatsapp/
│   ├── client.py              # neonize 客户端（QR 登录、收发、打字模拟）
│   └── debouncer.py           # 消息防抖
└── prompts/                   # Jinja2 模板（粤语为主）
    ├── system / safety / 各 task 模板
    └── agents/                # triage / safety_monitor / k6_scoring / memory
tools/
├── k6_query.py                # 查询单用户 K6 评分 + 决策日志
└── k6_export.py               # 批量导出 K6 评分到 CSV
tests/                         # 161 项单元测试
```

---

## 配置项（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **必填**，OpenRouter 密钥 |
| `OPENROUTER_MODEL` | `google/gemini-3.1-flash-lite` | 默认模型（各 Agent 留空时回退到它） |
| `MODEL_TRIAGE` / `MODEL_SAFETY` / `MODEL_K6` / `MODEL_THERAPIST` / `MODEL_MEMORY` | 空 | 各 Agent 模型档位；建议分诊/安全/K6/记忆用便宜模型，治疗师用质量好的 |
| `MEMORY_SUMMARY_EVERY` | `5` | 每隔几轮更新一次摘要 |
| `RECENT_TURNS_WITH_SUMMARY` | `4` | 有摘要时保留的原始轮数 |
| `RECENT_TURNS_NO_SUMMARY` | `8` | 触发摘要的阈值轮数 |
| `MIN_TURNS_PER_SCREENING` | `5` | 每个 K6/PM+ 阶段最少对话轮数 |
| `DEBOUNCE_SECONDS` | `3.0` | 消息防抖等待时间 |
| `RISK_ALPHA` / `BETA` / `GAMMA` / `DELTA` | `0.6 / 0.25 / 0.3 / 0.15` | R(t) 风险公式系数 |
| `MAX_TURNS` | `30` | 单次会话轮次上限 |

R(t) 公式：`R(t) = α·R(t-1) + β·s_emotion + γ·s_keyword + δ·s_behavior`，分四级 green(<30) / yellow(30-60) / orange(60-80) / red(≥80)。

---

## 查询与审计

```bash
# 查单个用户：K6 六维度分数、严重度、R(t)、PM+ 进度、决策审计日志
python tools/k6_query.py +85298765432

# 导出全部用户到 CSV（仅含哈希前缀，保护隐私）
python tools/k6_export.py
```

决策审计日志记录每次 PM+ 策略选择的依据（当时 K6 分数 + 选择 + 理由 + 时间），供复盘与合规追溯。

---

## 测试

```bash
python -m pytest tests/ -v
```

覆盖 FSM 转换、R(t) 公式、K6 评分器、PM+ 策略选择、各 Agent、Coordinator 调度、危机判定与锁定、记忆窗口、消息防抖、会话存储等。

---

## 危机资源（香港）

机器人在危机时会引导用户联系以下本地资源：

- 撒瑪利亞防止自殺會（24 小时）：**2389 2222**
- 醫院管理局精神健康專線（24 小时）：**2466 7350**
- 香港青年協會「關心一線」：**2777 8899**
- 香港明愛「向晴熱線」（24 小时）：**18288**
- 如有即时生命危险：**999**

---

## 重要声明

本机器人是情绪支援与筛查工具，**不能替代专业心理治疗或诊断**。所有 K6 评分为 LLM 辅助推断，仅供参考，不构成临床诊断。在识别危机信号时会引导用户联系上述本地专业资源。部署前请完成临床、伦理与法律审核。

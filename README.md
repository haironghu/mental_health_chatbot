# 心理健康 WhatsApp 聊天机器人（多 Agent 版本）

面向香港青少年的匿名心理健康支援聊天机器人。通过 WhatsApp 接入，使用 LLM 辅助 **K6 凯斯勒心理困扰量表**评估用户心理状态，并基于 **PM+（Problem Management Plus，WHO 设计）** 准则提供聊天安抚与干预。在识别危机信号时主动引导专业资源。

> 本仓库是在已交付的「单 LLM 调用版本」基础上，向**多 Agent 架构**演进的工作区。当前代码为单 LLM 起点，多 Agent 改造按 [docs/MULTI_AGENT_DESIGN.md](docs/MULTI_AGENT_DESIGN.md) 的路线图分阶段推进。

## 核心特性

- **K6 心理困扰评估**：LLM 在自然对话中推断 K6 六维度分数（紧张/无助/焦躁/抑郁/费力/无价值），跨轮平滑累积，输出 mild/moderate/severe 严重度
- **PM+ 干预策略**：根据 K6 结果选择对应策略 —— 管理压力（呼吸训练）、解决问题、行为激活、强化社交支持
- **实时风险监控 R(t)**：加权累积风险评分 + 四级预警（green/yellow/orange/red），独立于 K6，负责当下危机检测
- **规则驱动 FSM**：状态流转完全由规则决定，LLM 只负责自然语言生成，保证危机干预流程不被绕过
- **类人对话体验**：消息防抖、回复分段、打字延迟模拟、回复长度自适应
- **安全设计**：危机资源去重（避免重复推送热线）、偏题引导、药物咨询拦截、情绪稳定介入
- **隐私设计**：手机号 SHA-256 哈希存储，本地 JSON 存储，无身份资料收集

## 技术栈

- Python 3.14
- [neonize](https://github.com/krypton-byte/neonize)（whatsmeow WhatsApp Web 协议绑定）
- OpenRouter（OpenAI 兼容 API，当前 Claude Haiku）
- Jinja2（多层 Prompt 模板）
- pytest（132 项单元测试）

## 项目结构

```
app/
├── config.py                  # 配置（pydantic-settings）
├── agents/                    # ★ 多 Agent 架构（Phase 1-2）
│   ├── base.py                # Agent 基类 + AgentContext
│   ├── triage.py             # 分诊 Agent（情绪/行为信号 + 语言 + 意愿）
│   ├── safety_monitor.py     # 安全监测 Agent（危机/自伤/自杀检测）
│   ├── k6_scorer_agent.py    # K6 评分 Agent（仅 K6 阶段运行）
│   ├── therapist.py          # 治疗师 Agent（自然语言回复）
│   └── coordinator.py        # 协调器（并行调度 + 确定性决策）
├── orchestrator/
│   ├── fsm.py                 # 有限状态机（WELCOME→K6→PM+→CLOSURE）
│   └── orchestrator.py        # 会话生命周期（委托 Coordinator）
├── intelligence/
│   ├── llm.py                 # OpenRouter API 封装（支持按 agent 选模型）
│   └── prompt_builder.py      # 多层 Prompt 组装（System+Task+Safety）
├── safety/
│   ├── k6_scorer.py           # K6 评分器 + PM+ 策略选择
│   └── risk_monitor.py        # R(t) 风险公式 + 四级预警
├── storage/
│   └── session_store.py       # 本地 JSON 会话存储（手机号哈希）
├── whatsapp/
│   ├── client.py              # neonize 客户端（QR 登录、收发消息）
│   └── debouncer.py           # 消息防抖（合并连续短消息）
└── prompts/                   # Jinja2 模板（粤语为主）
    └── agents/                # 各 Agent 的 prompt（triage / k6_scoring）
tools/
├── k6_query.py                # 查询单用户 K6 评分
└── k6_export.py               # 批量导出 K6 评分到 CSV
tests/                         # 146 项单元测试
```

## 多 Agent 架构（Phase 1-2）

```
orchestrator.process()           # 会话生命周期
   └─> Coordinator.run()         # hub-and-spoke 调度
         ├─ 并行: TriageAgent + SafetyMonitorAgent + K6ScorerAgent（仅 K6 阶段）
         ├─ 三重危机判定: Safety LLM / 确定性关键词兜底 / R(t) 红色
         ├─ 确定性: R(t) 更新 / K6 更新 / FSM 决策 / 危机强制
         └─ TherapistAgent（生成回复）
```

设计原则、各 Agent 职责、后续 Phase 3-5 路线图详见 [docs/MULTI_AGENT_DESIGN.md](docs/MULTI_AGENT_DESIGN.md)。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 OPENROUTER_API_KEY

# 3. 启动（首次运行会显示 QR 码，用手机 WhatsApp 扫码登录）
python main.py
```

## 对话流程

```
WELCOME（欢迎）
   ↓
K6_ASSESSMENT（自然对话中评估 K6 六维度）
   ↓ K6 完成（≥4 维度有信号 + 聊够 5 轮）
PM+ 策略（根据 K6 严重度与维度选择）
   ↓ 聊够 5 轮
PM_DECISION（询问用户是否继续）
   ↓ 继续 → 下一策略 / 结束 → CLOSURE
CLOSURE（个性化告别）

CRISIS_INTERVENTION（任何时刻，R(t) red 或检测到危机意念时强制进入）
```

用户随时发送「開始」可重置会话。

## 查询 K6 评分

```bash
# 查单个用户（输入完整手机号，内部哈希查找）
python tools/k6_query.py +85298765432

# 导出全部用户到 CSV（仅含哈希前缀，保护隐私）
python tools/k6_export.py
```

## 测试

```bash
python -m pytest tests/ -v
```

## 重要声明

本机器人是情绪支援与筛查工具，**不能替代专业心理治疗或诊断**。在识别危机信号时会引导用户联系本地专业资源（如香港青年協會「關心一線」、撒瑪利亞防止自殺會等）。

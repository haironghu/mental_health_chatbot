from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemini-3.1-flash-lite"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7

    # 各 Agent 的模型档位（留空则回退 openrouter_model）
    # 建议：triage/k6 用便宜模型，therapist 用质量好的模型
    model_triage: str = ""       # 分诊（每轮跑，便宜）
    model_safety: str = ""       # 安全监测（每轮跑，便宜）
    model_k6: str = ""           # K6 评分（中档）
    model_therapist: str = ""    # 治疗师回复（质量优先）
    model_memory: str = ""       # 对话摘要（便宜）

    # 记忆 / 长会话摘要
    memory_summary_every: int = 5        # 每隔几轮更新一次摘要
    recent_turns_with_summary: int = 4   # 有摘要时保留嘅原始轮数
    recent_turns_no_summary: int = 8     # 无摘要时保留嘅原始轮数（触发摘要嘅阈值）

    # 每个筛查维度最少对话轮数（聊够才推进到下一个维度）
    min_turns_per_screening: int = 5

    # 消息防抖：收到消息后等待几秒，若期间有新消息则重置计时（让用户讲完）
    debounce_seconds: float = 3.0

    # 模拟打字：每个字符多少秒（如0.05秒/字 = 100字需要5秒）
    typing_seconds_per_char: float = 0.04
    # 单段最大延迟（避免过长消息让用户等太久）
    max_typing_delay_seconds: float = 4.0
    # 段间最小延迟
    min_typing_delay_seconds: float = 0.8

    # WhatsApp (neonize)
    whatsapp_db: str = "whatsapp_session.sqlite3"

    # 风险评分系数 R(t) = α×R(t-1) + β×S_emotion + γ×S_keyword + δ×S_behavior
    risk_alpha: float = 0.6
    risk_beta: float = 0.25
    risk_gamma: float = 0.3
    risk_delta: float = 0.15

    # 会话限制
    max_turns: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

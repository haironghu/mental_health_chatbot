"""
OpenRouter API 封装。
通过 OpenAI 兼容接口调用，当前使用 google/gemini-2.0-flash-exp:free。
"""
import json
import re

from openai import OpenAI

from app.config import settings

_client = OpenAI(
    base_url=settings.openrouter_base_url,
    api_key=settings.openrouter_api_key,
)


def complete(messages: list[dict], *, system: str = "") -> str:
    """
    调用 LLM 并返回文本回复。

    messages 格式：[{"role": "user"|"assistant", "content": "..."}]
    system   若非空则作为 system message 前置。
    """
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = _client.chat.completions.create(
        model=settings.openrouter_model,
        messages=full_messages,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    return response.choices[0].message.content


def complete_json(messages: list[dict], *, system: str = "") -> dict:
    """
    调用 LLM 并强制返回 JSON。
    通过 response_format 请求 JSON 输出，并做安全解析。
    """
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = _client.chat.completions.create(
        model=settings.openrouter_model,
        messages=full_messages,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    # 安全解析：提取第一个 JSON 对象
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM 未返回合法 JSON: {raw!r}")
    return json.loads(match.group())

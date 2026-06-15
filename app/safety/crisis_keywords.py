"""
确定性危机关键词检测。

作为 LLM Safety Monitor 的安全网：即使 LLM 调用失败或漏判，
明显的自伤/自杀表达也能被这层规则捕获，强制进入危机干预。

设计取舍：宁可偶尔误报（false positive）也唔可以漏报（false negative）。
危机干预本身係温和、支持性嘅，误触发嘅代价远低于漏判。
"""
import re

# 强信号短语（命中即视为危机）。涵盖粤语、普通话、英文。
_CRISIS_PATTERNS = [
    # 中文：自杀 / 不想活
    "想死", "想自殺", "自殺", "輕生", "结束生命", "結束生命",
    "结束自己", "結束自己", "唔想生存", "唔想活", "不想活",
    "不想活了", "活唔落去", "活不下去", "死咗算", "死了算",
    "不如死", "唔想再活", "了結自己", "了结自己", "我想消失",
    "想消失", "唔想再喺度", "生无可恋", "生無可戀",
    # 中文：自伤
    "自殘", "自残", "自我傷害", "自我伤害", "割脈", "割脉",
    "割手", "傷害自己", "伤害自己", "跳樓", "跳楼", "燒炭", "烧炭",
    # 英文
    "kill myself", "killing myself", "want to die", "wanna die",
    "end my life", "ending my life", "suicide", "suicidal",
    "self-harm", "self harm", "hurt myself", "hurting myself",
    "cut myself", "cutting myself", "no reason to live",
    "don't want to live", "dont want to live", "better off dead",
]

# 编译为大小写不敏感的正则（英文部分需要；中文不受影响）
_COMPILED = [re.compile(re.escape(p), re.IGNORECASE) for p in _CRISIS_PATTERNS]


def contains_crisis_keywords(text: str) -> bool:
    """文本中是否包含明确的危机关键词。"""
    if not text:
        return False
    return any(pat.search(text) for pat in _COMPILED)


def matched_keywords(text: str) -> list[str]:
    """返回命中的关键词列表（用于日志 / 审计）。"""
    if not text:
        return []
    return [pat.pattern.replace("\\", "") for pat in _COMPILED if pat.search(text)]

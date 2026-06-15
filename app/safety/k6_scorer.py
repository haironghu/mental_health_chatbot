"""
K6 凱斯勒心理困擾量表評分器。

K6 六維度（每題 0-4 分）：
- tense       緊張
- helpless    無助
- restless    焦躁 / 坐立不安
- depressed   抑鬱沉重，乜都唔開心
- effortful   所有嘢都好費力
- worthless   覺得自己冇價值

0 = 從不   1 = 很少   2 = 有時   3 = 頗多   4 = 總是

總分 0-24：
- 0-4   mild      輕度困擾
- 5-12  moderate  中度困擾
- 13-24 severe    重度困擾

設計要點：
- 跨輪平滑：新分 = max(舊分, LLM 本輪推斷分)
  → K6 設計上測「過去 4 週」，所以分數應該只升不降，符合「累積證據」嘅語義
- 完成判定：≥4 / 6 維度有非零信號 + state_turn_count ≥ 5 轮
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

K6_DIMENSIONS = (
    "tense", "helpless", "restless",
    "depressed", "effortful", "worthless",
)

K6_LABELS_ZH = {
    "tense": "緊張",
    "helpless": "無助",
    "restless": "焦躁",
    "depressed": "抑鬱",
    "effortful": "費力",
    "worthless": "無價值",
}

K6_RESPONSE_LABELS = {
    0: "從不",
    1: "很少",
    2: "有時",
    3: "頗多",
    4: "總是",
}

# 完成判定参数
_MIN_NONZERO_DIMS = 4   # 至少 4 个维度有非零信号
_MIN_TURNS = 5          # K6_ASSESSMENT 状态至少聊 5 轮


@dataclass
class K6Result:
    scores: dict[str, int]    # 各维度分数 0-4
    total: int                # 0-24
    severity: str             # mild / moderate / severe
    complete: bool            # 是否已完成评估


def severity_of(total: int) -> str:
    """根据 K6 总分返回严重度等级。"""
    if total <= 4:
        return "mild"
    if total <= 12:
        return "moderate"
    return "severe"


def update_scores(session: dict, llm_dim_scores: dict[str, int]) -> dict[str, int]:
    """
    跨轮平滑：每个维度取 max(旧分, LLM 本轮新推断)。
    返回更新后的 k6_scores（已原地写入 session）。
    """
    current = dict(session.get("k6_scores", {dim: 0 for dim in K6_DIMENSIONS}))
    for dim in K6_DIMENSIONS:
        old = int(current.get(dim, 0))
        new = int(llm_dim_scores.get(dim, 0))
        # clamp 到 0-4
        new = max(0, min(4, new))
        current[dim] = max(old, new)
    session["k6_scores"] = current
    return current


def evaluate(session: dict, state_turn_count: int) -> K6Result:
    """
    根据 session 当前的 k6_scores 计算总分、严重度、完成判定。
    完成判定：≥4 个维度非零 + 已聊够 _MIN_TURNS 轮。
    """
    scores = session.get("k6_scores", {dim: 0 for dim in K6_DIMENSIONS})
    total = sum(int(scores.get(dim, 0)) for dim in K6_DIMENSIONS)
    severity = severity_of(total)
    nonzero_dims = sum(1 for dim in K6_DIMENSIONS if int(scores.get(dim, 0)) > 0)
    complete = nonzero_dims >= _MIN_NONZERO_DIMS and state_turn_count >= _MIN_TURNS
    return K6Result(
        scores=dict(scores),
        total=total,
        severity=severity,
        complete=complete,
    )


def mark_complete(session: dict, result: K6Result) -> None:
    """将 K6 评估结果写回 session（标记完成）。"""
    session["k6_total"] = result.total
    session["k6_severity"] = result.severity
    session["k6_complete"] = True
    session["k6_completed_at"] = datetime.now().isoformat(timespec="seconds")


# ────────────────────────────────────────────────────────────────
# PM+ 策略选择：根据 K6 维度高分映射到对应策略
# ────────────────────────────────────────────────────────────────

# 维度 → 推荐 PM+ 策略
_DIMENSION_STRATEGY = {
    "tense": "pm_stress_mgmt",
    "restless": "pm_stress_mgmt",
    "helpless": "pm_behavioral_activation",
    "depressed": "pm_behavioral_activation",
    "effortful": "pm_behavioral_activation",
    "worthless": "pm_social_support",
}

# 所有 PM+ 策略，回退顺序
_ALL_STRATEGIES = [
    "pm_stress_mgmt",
    "pm_problem_solving",
    "pm_behavioral_activation",
    "pm_social_support",
]


def select_next_pm_strategy(
    k6_scores: dict[str, int],
    used: list[str],
) -> Optional[str]:
    """
    根据 K6 各维度分数和已使用策略，选下一个最合适嘅 PM+ 策略。

    优先级：
      1. 按各维度分数从高到低排序，第一个对应嘅策略（如未用过）
      2. 若多个维度高分模糊 → pm_problem_solving
      3. 所有专属策略都用过 → 返回 None（结束）

    返回策略状态名（如 "pm_stress_mgmt"），或 None 表示无可推荐。
    """
    # 按分数排序维度（高到低）
    ranked = sorted(
        K6_DIMENSIONS,
        key=lambda d: int(k6_scores.get(d, 0)),
        reverse=True,
    )

    # 第一步：尝试根据最高分维度映射的策略
    for dim in ranked:
        score = int(k6_scores.get(dim, 0))
        if score == 0:
            break  # 后面都是 0，没必要再看
        strategy = _DIMENSION_STRATEGY.get(dim)
        if strategy and strategy not in used:
            return strategy

    # 第二步：如果多个维度都有非零但策略都用过，用 problem_solving 兜底
    if "pm_problem_solving" not in used:
        # 只在用户实际有困扰时才推荐解决问题
        if any(int(k6_scores.get(d, 0)) >= 2 for d in K6_DIMENSIONS):
            return "pm_problem_solving"

    # 第三步：所有策略都用过 → 结束
    return None

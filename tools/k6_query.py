"""
单用户 K6 查询工具。

用法：
    python tools/k6_query.py +85298765432

输入完整手机号（带或不带 +），工具内部哈希后查找 session。
不会反向查询哈希→手机号，保护隐私。
"""
import argparse
import sys
from pathlib import Path

# 允许从项目根目录运行
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.safety.k6_scorer import K6_DIMENSIONS, K6_LABELS_ZH, K6_RESPONSE_LABELS  # noqa: E402
from app.storage import session_store  # noqa: E402


def query(phone: str) -> None:
    user_hash = session_store._user_hash(phone)
    session = session_store.load(user_hash)

    if session is None:
        print(f"未找到该手机号的会话记录")
        print(f"(查询的 hash 前缀: {user_hash[:8]}...)")
        sys.exit(1)

    print("=" * 50)
    print(f"用户哈希前缀: {user_hash[:8]}...")
    print(f"会话状态:    {session.get('state')}")
    print(f"轮次:        {session.get('turn_count')}")
    print()

    # K6 评估
    print("--- K6 凱斯勒心理困擾量表 ---")
    if session.get("k6_complete"):
        print(f"评估状态:  已完成")
        print(f"完成时间:  {session.get('k6_completed_at') or '未记录'}")
    else:
        print("评估状态:  进行中（尚未完成）")
    print()
    print("六维度分数:")
    scores = session.get("k6_scores", {})
    for dim in K6_DIMENSIONS:
        score = int(scores.get(dim, 0))
        label = K6_LABELS_ZH[dim]
        resp = K6_RESPONSE_LABELS.get(score, "?")
        print(f"  {label:6}  {score}  ({resp})")
    print()
    total = session.get("k6_total", sum(int(scores.get(d, 0)) for d in K6_DIMENSIONS))
    severity = session.get("k6_severity", "?")
    print(f"总分: {total} / 24")
    print(f"严重度: {severity}")
    print()

    # 实时风险
    print("--- 实时风险监控（R(t)） ---")
    print(f"R(t) 分数:   {session.get('risk_score', 0):.1f}")
    print(f"预警级别:    {session.get('alert_level', 'green')}")
    print(f"连续负面轮次: {session.get('consecutive_negative_turns', 0)}")
    print()

    # PM+ 进度
    print("--- PM+ 策略 ---")
    used = session.get("pm_strategies_used", [])
    if used:
        labels = {
            "pm_stress_mgmt": "管理压力（呼吸训练）",
            "pm_problem_solving": "解决问题",
            "pm_behavioral_activation": "行为激活",
            "pm_social_support": "强化社交支持",
        }
        for s in used:
            print(f"  ✓ {labels.get(s, s)}")
    else:
        print("  暂未使用过 PM+ 策略")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查询单个用户的 K6 评估结果")
    parser.add_argument("phone", help="用户手机号（带或不带 +，例如 +85298765432 或 85298765432）")
    args = parser.parse_args()
    query(args.phone)

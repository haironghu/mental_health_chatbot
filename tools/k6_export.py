"""
批量导出全部用户的 K6 评分到 CSV。

用法：
    python tools/k6_export.py
    python tools/k6_export.py --output data/k6_export.csv
    python tools/k6_export.py --only-complete

CSV 只包含用户哈希前缀（不暴露原始手机号）。
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.safety.k6_scorer import K6_DIMENSIONS  # noqa: E402
from app.storage import session_store  # noqa: E402


def export(output_path: Path, only_complete: bool = False) -> int:
    sessions_dir = session_store.SESSIONS_DIR
    rows = []

    for session_file in sorted(sessions_dir.glob("*.json")):
        user_hash = session_file.stem  # 文件名即 hash
        try:
            import json
            with session_file.open(encoding="utf-8") as f:
                session = json.load(f)
        except Exception as e:
            print(f"跳过 {session_file.name}: {e}", file=sys.stderr)
            continue

        if only_complete and not session.get("k6_complete"):
            continue

        scores = session.get("k6_scores", {})
        row = {
            "user_hash_prefix": user_hash[:8],
            "state": session.get("state", ""),
            "k6_complete": session.get("k6_complete", False),
            "k6_completed_at": session.get("k6_completed_at") or "",
            "k6_total": session.get("k6_total", 0),
            "k6_severity": session.get("k6_severity", ""),
        }
        for dim in K6_DIMENSIONS:
            row[f"k6_{dim}"] = int(scores.get(dim, 0))
        row["risk_score"] = round(session.get("risk_score", 0), 1)
        row["alert_level"] = session.get("alert_level", "")
        row["pm_strategies_used"] = "|".join(session.get("pm_strategies_used", []))
        row["turn_count"] = session.get("turn_count", 0)
        rows.append(row)

    if not rows:
        print("没有可导出的数据")
        return 0

    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已导出 {len(rows)} 条记录到 {output_path}")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出全部用户的 K6 评分到 CSV")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/k6_export.csv"),
        help="输出 CSV 路径（默认 data/k6_export.csv）",
    )
    parser.add_argument(
        "--only-complete",
        action="store_true",
        help="只导出已完成 K6 评估的用户",
    )
    args = parser.parse_args()
    export(args.output, args.only_complete)

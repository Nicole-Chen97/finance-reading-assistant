"""Router 分類準確率評估腳本。

跟 tests/ 底下的單元測試不一樣：這裡**真的會呼叫 Anthropic API**（不 mock router_llm），
目的是回答「router 這個 LLM 面對真實使用者問題，分類準不準」，這件事沒辦法用 mock 測出來。

用法：
    cd Substack_post_download
    uv run python evals/router/eval_router.py
    uv run python evals/router/eval_router.py --threshold 0.9   # 核心案例準確率低於這個值時 exit code 非 0
    uv run python evals/router/eval_router.py --set evals/router/golden_set.jsonl

不會被 `pytest tests/` 收集到、也不在 CI 裡自動跑——會實際花 API 額度、每題都是一次真的 LLM 呼叫，
故意讓它是「需要手動執行」的評估腳本，而不是每次 commit 都跑的測試。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from langchain_core.messages import HumanMessage  # noqa: E402
import graph as graph_module  # noqa: E402  (真正的 router_llm，不 mock)

ROUTES = ["download", "rag", "market_price", "chat"]


def load_golden_set(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def classify(query: str) -> str:
    """直接呼叫真正的 router_node，跟 production 走同一段程式碼（同一份 prompt），
    避免評估腳本自己重寫一份 prompt 之後跟實際邏輯long-term跑偏。"""
    state = {"messages": [HumanMessage(content=query)], "router": None, "tool_result": None}
    result = graph_module.router_node(state)
    return result["router"]


def run(golden_set_path: Path) -> list[dict]:
    cases = load_golden_set(golden_set_path)
    results = []
    for i, case in enumerate(cases, 1):
        predicted = classify(case["query"])
        correct = predicted == case["expected_route"]
        results.append({**case, "predicted_route": predicted, "correct": correct})
        status = "OK  " if correct else "MISS"
        print(f"[{i}/{len(cases)}] {status} {case['id']:8s} expected={case['expected_route']:13s} "
              f"predicted={predicted:13s} | {case['query']}")
    return results


def summarize(results: list[dict]) -> dict:
    core = [r for r in results if not r["ambiguous"]]
    ambiguous = [r for r in results if r["ambiguous"]]

    def accuracy(rs):
        return (sum(r["correct"] for r in rs) / len(rs)) if rs else None

    confusion = defaultdict(Counter)  # confusion[expected][predicted] += 1
    for r in results:
        confusion[r["expected_route"]][r["predicted_route"]] += 1

    per_route = {
        route: accuracy([r for r in core if r["expected_route"] == route])
        for route in ROUTES
    }

    return {
        "n_total": len(results),
        "n_core": len(core),
        "n_ambiguous": len(ambiguous),
        "core_accuracy": accuracy(core),
        "ambiguous_accuracy": accuracy(ambiguous),
        "per_route_accuracy": per_route,
        "confusion": {e: dict(p) for e, p in confusion.items()},
        "misses_core": [r for r in core if not r["correct"]],
        "misses_ambiguous": [r for r in ambiguous if not r["correct"]],
    }


def print_report(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("Router 分類準確率報告")
    print("=" * 60)
    print(f"核心案例（不含 ambiguous）: {summary['core_accuracy']:.1%} "
          f"({summary['n_core'] - len(summary['misses_core'])}/{summary['n_core']})")
    if summary["n_ambiguous"]:
        print(f"邊界案例（ambiguous，僅供參考，不計入核心指標）: "
              f"{summary['ambiguous_accuracy']:.1%} ({summary['n_ambiguous'] - len(summary['misses_ambiguous'])}/{summary['n_ambiguous']})")

    print("\n各路線準確率（僅核心案例）：")
    for route, acc in summary["per_route_accuracy"].items():
        print(f"  {route:13s}: {'n/a' if acc is None else f'{acc:.1%}'}")

    print("\n混淆矩陣（expected -> predicted 次數）：")
    for expected in ROUTES:
        row = summary["confusion"].get(expected, {})
        row_str = ", ".join(f"{p}={c}" for p, c in row.items())
        print(f"  {expected:13s}: {row_str}")

    if summary["misses_core"]:
        print("\n核心案例分類錯誤明細：")
        for r in summary["misses_core"]:
            print(f"  [{r['id']}] \"{r['query']}\"\n"
                  f"      expected={r['expected_route']}  predicted={r['predicted_route']}")

    if summary["misses_ambiguous"]:
        print("\n邊界案例「不同意」明細（不代表 router 錯，只是跟我主觀認定的答案不同）：")
        for r in summary["misses_ambiguous"]:
            print(f"  [{r['id']}] \"{r['query']}\"\n"
                  f"      my_label={r['expected_route']}  predicted={r['predicted_route']}  note={r['note']}")


def save_report(results: list[dict], summary: dict, out_path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {k: v for k, v in summary.items() if k not in ("misses_core", "misses_ambiguous")},
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果已存到: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path(__file__).parent / "golden_set.jsonl")
    parser.add_argument("--threshold", type=float, default=0.9,
                         help="核心案例準確率低於這個值時，腳本以非 0 exit code 結束（方便接 CI gate）")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "last_run_report.json")
    args = parser.parse_args()

    results = run(args.set)
    summary = summarize(results)
    print_report(summary)
    save_report(results, summary, args.out)

    if summary["core_accuracy"] is not None and summary["core_accuracy"] < args.threshold:
        print(f"\n❌ 核心準確率 {summary['core_accuracy']:.1%} 低於門檻 {args.threshold:.0%}")
        sys.exit(1)
    print(f"\n✅ 核心準確率 {summary['core_accuracy']:.1%} 達到門檻 {args.threshold:.0%}")


if __name__ == "__main__":
    main()

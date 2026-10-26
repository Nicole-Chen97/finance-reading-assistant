"""查價參數抽取評估：LLM 把一句話轉成的 Alpaca 查詢參數，對不對？

只評估 market_tool.extract_query_params（LLM 抽參數）這一步，**不會真的去打 Alpaca**：
參數抽對了，後面查價、畫圖都是固定的程式邏輯（有單元測試）；參數抽錯了，
系統會「正常地」查到錯的股票或錯的日期，畫面看起來一切正常，使用者很難發現 —— 這正是需要評估的原因。

逐欄位計分，答錯時才知道錯在哪裡：
  category  資產類別（stock / crypto）
  symbols   代號（不分大小寫、不管順序；有些題目接受替代答案，例如 GOOGL / GOOG）
  mode      即時（latest）還是歷史（history）
  fig       要不要畫圖（只有題目明確要求或明確沒要求時才計分）
  start/end 歷史區間的起訖日。題庫用「幾天前」這種相對日期，跑的當天自動換算，所以哪天跑都能用

跟 eval_router.py 一樣會真的呼叫 Anthropic API（每題一次），需要手動執行。

用法：
    cd Substack_post_download
    uv run python evals/market/eval_market_params.py
"""
import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

from market_tool import extract_query_params  # noqa: E402

FIELDS = ["category", "symbols", "mode", "fig", "start", "end"]


def resolve_date(spec: dict, today: date) -> date:
    if "days_ago" in spec:
        return today - timedelta(days=spec["days_ago"])
    if spec["date"] == "THIS_YEAR_START":
        return date(today.year, 1, 1)
    if spec["date"] == "LAST_MONTH_START":
        first_of_this_month = today.replace(day=1)
        return (first_of_this_month - timedelta(days=1)).replace(day=1)
    return date.fromisoformat(spec["date"])


def parse_param_date(value) -> date | None:
    if not value:
        return None
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def check_date(actual, specs: list[dict], today: date) -> tuple[bool, str]:
    actual_date = parse_param_date(actual)
    if actual_date is None:
        return False, "沒有日期"
    for spec in specs:
        expected = resolve_date(spec, today)
        if abs((actual_date - expected).days) <= spec["tol"]:
            return True, ""
    expected_str = " 或 ".join(str(resolve_date(s, today)) for s in specs)
    return False, f"預期約 {expected_str}"


def normalize_symbols(value: str) -> frozenset:
    return frozenset(s.strip().upper() for s in str(value).replace(" ", ",").split(",") if s.strip())


def score(case: dict, params, today: date) -> dict:
    """回傳每個欄位 {"ok": bool, "actual": ..., "why": ...}；題目沒要求的欄位不列入。"""
    result = {}
    result["category"] = {"ok": params.category == case["category"], "actual": params.category}

    actual_symbols = normalize_symbols(params.symbols)
    accepted = [frozenset(case["symbols"])] + [frozenset(alt) for alt in case.get("symbols_alt", [])]
    result["symbols"] = {"ok": actual_symbols in accepted, "actual": params.symbols}

    result["mode"] = {"ok": params.mode == case["mode"], "actual": params.mode}

    if "fig" in case:
        result["fig"] = {"ok": bool(params.fig) == case["fig"], "actual": params.fig}

    for key in ("start", "end"):
        if key in case:
            ok, why = check_date(getattr(params, key), case[key], today)
            result[key] = {"ok": ok, "actual": str(getattr(params, key))[:10], "why": why}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", type=Path, default=Path(__file__).parent / "golden_set.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "last_run_report.json")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    cases = [json.loads(line) for line in args.set.open(encoding="utf-8") if line.strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        try:
            params = extract_query_params(case["query"])
            fields = score(case, params, today)
            error = None
        except Exception as e:  # LLM 輸出不合格式、validator 擋下來等，都算這題錯
            fields, error = {}, f"{type(e).__name__}: {e}"
        all_ok = error is None and all(f["ok"] for f in fields.values())
        rows.append({**case, "fields": fields, "error": error, "all_ok": all_ok})
        wrong = ", ".join(f"{k}={v['actual']}{'（' + v['why'] + '）' if v.get('why') else ''}"
                          for k, v in fields.items() if not v["ok"]) or error or ""
        print(f"[{i}/{len(cases)}] {'OK  ' if all_ok else 'MISS'} {case['id']:7s} {case['query']}"
              + (f"\n          錯在：{wrong}" if not all_ok else ""), flush=True)

    core = [r for r in rows if not r["ambiguous"]]
    field_acc = {}
    for f in FIELDS:
        scored = [r["fields"][f]["ok"] for r in core if f in r["fields"]]
        if scored:
            field_acc[f] = {"accuracy": sum(scored) / len(scored), "n": len(scored)}
    summary = {
        "today": str(today),
        "n_core": len(core),
        "all_fields_correct": sum(r["all_ok"] for r in core) / len(core),
        "per_field": field_acc,
        "ambiguous_all_correct": sum(r["all_ok"] for r in rows if r["ambiguous"]),
        "n_ambiguous": len(rows) - len(core),
    }

    print("\n" + "=" * 60)
    print(f"查價參數抽取評估（核心題 {len(core)} 題，評估日 {today}）")
    print("=" * 60)
    print(f"所有欄位全對：{summary['all_fields_correct']:.0%}")
    for f, v in field_acc.items():
        print(f"  {f:9s}: {v['accuracy']:.0%}  (n={v['n']})")
    print(f"邊界題全對：{summary['ambiguous_all_correct']}/{summary['n_ambiguous']}（僅供參考）")

    args.out.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    print(f"\n完整結果已存到: {args.out}")


if __name__ == "__main__":
    main()

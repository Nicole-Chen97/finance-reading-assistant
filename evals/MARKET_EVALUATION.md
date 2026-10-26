# 查價參數抽取評估

## 為什麼要評估這一步

查價流程是：**LLM 把一句話轉成查詢參數** → 用參數去打 Alpaca → 產生報告和 K 線圖。

後面兩步是固定的程式邏輯，已經有單元測試。第一步是 LLM 在判斷，而且**錯了很難發現**：如果 LLM 把「輝達」轉成錯的代號，或把「最近一個月」算成兩年前，系統會正常地查到資料、畫出圖，畫面看起來一切正常，只是內容是錯的。

## 測什麼

| 欄位 | 例子 | 可能錯在哪裡 |
|---|---|---|
| `category` | 股票 `stock` / 加密貨幣 `crypto` | 「SOL」只給代號，要判斷是加密貨幣 |
| `symbols` | `TSLA`、`BTC/USD` | 中文名 → 代號（特斯拉、輝達、狗狗幣）；加密貨幣要帶 `/USD`；新上市的公司（Circle → `CRCL`） |
| `mode` | 即時 `latest` / 歷史 `history` | 「走勢」「變化」是歷史，「現在多少」是即時 |
| `fig` | 要不要畫 K 線圖 | 使用者說「畫圖」「K 線」才要畫 |
| `start` / `end` | 歷史區間 | **相對日期**：「最近一個月」「今年以來」需要知道今天是幾號 |

- 題庫：[market/golden_set.jsonl](market/golden_set.jsonl)，24 題（核心 22 題 + 邊界 2 題）。日期用「幾天前」這種相對寫法，哪天跑都能自動換算。
- 腳本：[market/eval_market_params.py](market/eval_market_params.py)。只評估 [market_tool.py](../app/market_tool.py) 的 `extract_query_params()`，**不會真的打 Alpaca**。
- 每題一次 Anthropic API 呼叫，24 題的費用很低。

```bash
cd Substack_post_download
uv run python evals/market/eval_market_params.py
```

## 狀態：⏳ 等 API 額度補上後執行

計分邏輯已經用假的 LLM 輸出驗證過（2026-10-25）：替代答案、大小寫、代號順序、少了 `/USD`、年份猜錯、「上個月」的兩種解讀，都能正確判斷。

## 讀程式碼時發現、還沒驗證的問題

照「先量再改」的原則，**這些都先不改**。等基準線跑出來，確認問題真的存在、影響多大，再修正並重測，作品集上才會有修正前後的對比。

| # | 疑點 | 位置 | 預期影響 | 如果確認了，怎麼修 |
|---|---|---|---|---|
| 1 | **LLM 不知道今天的日期**：prompt 裡只有使用者的問題，沒有告訴 LLM 今天幾號 | `extract_query_params()` | 「最近一個月」「今年以來」這類題目，日期可能填成 LLM 訓練資料時期的年份（his-02、his-03、his-04、his-07、his-09） | 在 system prompt 加上「今天是 YYYY-MM-DD」 |
| 2 | **抽參數用 temperature 0.7**：抽參數是「有標準答案」的工作，通常用 0 | 同上 | 同一句話重問，可能得到不同的參數，結果不穩定 | 改成 temperature 0；可以重跑同一份題庫 3 次，比較答案一不一致 |
| 3 | 沒有 system prompt 說明規則：例如「沒說期間就查 90 天」「說畫圖才畫」 | 同上 | 只能靠欄位說明（`Field(description=...)`），LLM 有時會忽略 | 加一段簡短的 system prompt |

另外有一個設計限制，不算 bug：Alpaca 只有美股和加密貨幣，使用者問「台積電」時只能查美股 ADR（`TSM`），沒辦法查台股 2330。題庫裡把這題標成邊界題。

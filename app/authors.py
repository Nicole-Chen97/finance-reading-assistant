"""作者設定：從 JSON 設定檔讀取，程式碼裡不寫死任何作者名稱。

讀取順序：
1. 環境變數 AUTHORS_CONFIG 指定的檔案（測試用這個指向範例檔）
2. 專案根目錄的 authors.json（真正的設定，不進版控）
3. 專案根目錄的 authors.example.json（範本，用假名稱）

設定檔格式見 authors.example.json：
- name：資料庫 metadata 裡 author 欄位存的值（Substack 文章的作者名稱），作者 filter 用它比對
- aliases：使用者可能怎麼稱呼這位作者，寫進 prompt 讓 LLM 對應到 name
- profile_user_id：Substack 的公開數字 ID，下載文章用
- sid_env：存放這位作者登入 cookie 的環境變數名稱（cookie 是機密，放 .env）
"""
import json
import os
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _config_path() -> Path:
    if os.getenv("AUTHORS_CONFIG"):
        return Path(os.environ["AUTHORS_CONFIG"])
    real = PROJECT_ROOT / "authors.json"
    return real if real.exists() else PROJECT_ROOT / "authors.example.json"


AUTHORS: list[dict] = json.loads(_config_path().read_text(encoding="utf-8"))
AUTHOR_NAMES: tuple[str, ...] = tuple(a["name"] for a in AUTHORS)

# 用 Literal 而不是 str：LLM 收到的 JSON schema 裡只有這幾個選項，
# 從格式層面杜絕填出資料庫裡沒有的名稱（之前作者 filter 從沒生效過就是因為名稱拼法對不上）
AuthorName = Literal[AUTHOR_NAMES]


def author_prompt_hint() -> str:
    """給 prompt 用的作者對照說明，例如：- 'Author A'：提到 Author A、A 作者"""
    return "\n".join(
        f"- '{a['name']}'：提到 {'、'.join([a['name'], *a.get('aliases', [])])}"
        for a in AUTHORS
    )


def get_author(name: str) -> dict | None:
    return next((a for a in AUTHORS if a["name"] == name), None)

import os
import sys
from pathlib import Path

# app/ 裡面的模組互相用 flat import（例如 `from tool import ...`），
# 不是 `from app.tool import ...`，所以要把 app/ 這個資料夾加進 sys.path，
# 測試檔案才能 `from schema import ...`、`from market_tool import ...`。
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

# 測試一律用 authors.example.json 的假作者（Author A、Author B），
# 不受本機 authors.json 影響，也不讓真正的作者名稱出現在測試裡。
os.environ["AUTHORS_CONFIG"] = str(APP_DIR.parent / "authors.example.json")

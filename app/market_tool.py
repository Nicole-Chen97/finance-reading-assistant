import logging
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta



import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)



HEADERS = {
    "accept": "application/json",
    "APCA-API-KEY-ID": os.environ["APCA-API-KEY-ID"],
    "APCA-API-SECRET-KEY": os.environ["APCA-API-SECRET-KEY"]
}




#==================AlpacaQueryParams======================
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
#... : 必填
class AlpacaQueryParams(BaseModel):
    # 只列出 URL_MAP 裡真的有實作的類別（option/forex 目前沒有對應的 Alpaca API 端點）——
    # 選項要跟實作同步，不然 LLM 會選中一個「合法但沒實作」的類別，導致查詢失敗。
    category: Literal["stock", "crypto"] = Field(
        ..., description="資產類別。股票、ETF 選 'stock'，加密貨幣選 'crypto'。"
    )
    
    symbols: str = Field(
        ..., description="代號。股票如 'AAPL'；加密貨幣如 'BTC/USD' (必須包含 /USD)。"
    )
    
    mode :Optional[Literal["history","latest"]] = Field(
       default =  "latest", 
        description="查詢模式：'history' (歷史區間數據) 或 'latest' (最新數據)"
    )
    
    
    

    # 歷史數據bar 專用欄位，設為 Optional
    timeframe: Optional[str] = Field(default="1Day", description="資料頻率，如 '1Min', '1H', '1Day', '1W', '1M'")
    start: Optional[str] = Field(None, description="開始日期 (YYYY-MM-DD)。若為 None，預設為 90 天前。")
    end: Optional[str] = Field(None, description="結束日期 (YYYY-MM-DD)。若為 None，預設為今天。")
    
    
    # 宣告 feed 為 Optional，因為 crypto 不一定需要
    feed: Optional[str] = Field(None, description="資料來源，stock 預設為 iex")
    fig : Optional[bool] = Field(None, description="是否出圖。")
    
    # mode='after' 的意思是在「所有預設邏輯跑完之後」再執行你的函數
    @model_validator(mode='after')
    def validate_logic(self):
        # 處理 Feed 邏輯 (只在類別為 stock 時必要)
        if self.category == "stock" and self.feed is None:
            self.feed = "iex"
        
        if self.mode =="latest":
            self.timeframe =None
            self.fig =False
            
        # 處理預設時間 (如果 is_latest 為 False)，如果要找歷史數據，或是一段時間區間數據
        if self.mode == "history":
            if self.fig is None:
                self.fig = False # 預設不出圖
                
                
            
            end_val = self.end if self.end else datetime.now()
            if isinstance(end_val, str): 
                end_val = datetime.strptime(end_val, '%Y-%m-%d')
            
            # 直接將「處理好的字串」賦值給 self.end
            self.end = end_val.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            
            if isinstance(self.start, str): 
                start_val = datetime.strptime(self.start, '%Y-%m-%d')
                self.start =  start_val.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            else :
                self.start = (end_val - timedelta(days=90)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
        return self   
     
    
    
# ================fetch_alpaca_data==================
URL_MAP = {
    "stock" :{
        "history" : "https://data.alpaca.markets/v2/stocks/bars" , 
        "latest" : "https://data.alpaca.markets/v2/stocks/bars/latest",
        },
    "crypto":{
        "history" : "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        "latest": "https://data.alpaca.markets/v1beta3/crypto/us/latest/bars"
    }  
}

def fetch_alpaca_data(params: AlpacaQueryParams):
    """
    根據 Pydantic 模型參數自動獲取 Alpaca 市場數據
    """
    try:
        target_url = URL_MAP[params.category][params.mode]
        payload = params.model_dump(exclude={'category', 'mode','fig'})
        response = requests.get(target_url, params=payload, headers=HEADERS)
        
        # 檢查 HTTP 狀態碼
        response.raise_for_status()
        data = response.json()
        
        return data
    
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP 錯誤: {http_err}"}
    except Exception as err:
        return {"error": f"發生意外錯誤: {err}"} 
        


# ================generate_market_data==================
def generate_market_report(result, symbols, mode):
    data = result.get('bars', {})
    
    # 處理 latest 模式
    if mode == 'latest':
        # 這裡處理 Alpaca latest 可能回傳單筆物件或列表的情況
        price_info = data.get(symbols)
        if isinstance(price_info, list):
            price_info = price_info[0]
            
        report = (
            f"[{symbols} 最新報價]\n"
            f"時間：{price_info['t']}\n"
            f"成交量加權平均價 (VWAP)：{price_info['vw']}\n"
            f"收盤價：{price_info['c']}\n"
            f"成交量：{price_info['v']}"
        )
        return report

    # 處理 history 模式
    elif mode == 'history':
        # 確保資料轉為 DataFrame 並處理單筆資料問題
        raw_data = data.get(symbols, [])
        df = pd.DataFrame(raw_data) if isinstance(raw_data, list) else pd.DataFrame([raw_data])

        df['t'] = pd.to_datetime(df['t'])
        # 降序排列，最新的在第 0 行
        df = df.sort_values(by='t', ascending=False)

        # history 模式代表的是「一段區間」，不是單一天，摘要文字要把整段區間的
        # 起訖日期、區間最高/最低、整體漲跌幅都講清楚，不能只描述最新一天的數字——
        # 不然 LLM 會誤以為只查得到一天的資料，寫出跟圖表（用的是完整區間資料）矛盾的回答。
        latest = df.iloc[0]
        earliest = df.iloc[-1]

        period_high = df['h'].max()
        period_low = df['l'].min()
        change_pct = (latest['c'] - earliest['c']) / earliest['c'] * 100 if earliest['c'] else 0.0

        report = (
            f"[{symbols} 歷史數據摘要]\n"
            f"區間: {earliest['t'].strftime('%Y年%m月%d日')} ~ {latest['t'].strftime('%Y年%m月%d日')}（共 {len(df)} 筆）\n"
            f"最新收盤價: ${latest['c']:,.2f}\n"
            f"區間最高: ${period_high:,.2f}\n"
            f"區間最低: ${period_low:,.2f}\n"
            f"區間漲跌幅: {change_pct:+.2f}%\n"
            f"最新成交量: {latest['v']:.4f}"
        )
        return report
    


# ================plot_candlestick==================
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_candlestick(json_data, symbol,timeframe):
    # 1. 將 API 返回的 JSON 解析為 DataFrame
    # 假設 json_data 是 response.json() 的結果
    bars = json_data['bars'].get(symbol)
    if not bars:
        raise ValueError(f"No data found for symbol: {symbol}")
    
    df = pd.DataFrame(bars)
    
    # 2. 清理與轉換時間格式 (Alpaca 的時間是 ISO 格式)
    df['t'] = pd.to_datetime(df['t'])
    
    # df['EMA5'] = df['c'].rolling(window=5).mean()
    df['EMA5'] = df['c'].ewm(span=5, adjust=False).mean()
    
    df['EMA20'] = df['c'].ewm(span=20, adjust=False).mean() 
    df['EMA10'] = df['c'].ewm(span=10, adjust=False).mean() 
    
    
    
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, 
                    subplot_titles=(f'{symbol} Price', 'Volume'),
                    row_width=[0.3, 0.7]) # 比例：成交量佔 20%，價格佔 70%
    
    # 3. 繪製蠟燭圖
    fig.add_trace(go.Candlestick(
        x=df['t'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        name='Price'), row=1, col=1)
    
    # 繪製均線 (疊加在 Row 1)
    colors = {'EMA5': 'orange', 'EMA10': 'cyan', 'EMA20': 'magenta'}
    for ma in ['EMA5', 'EMA10', 'EMA20']:
        fig.add_trace(go.Scatter(
            x=df['t'], y=df[ma], 
            line=dict(color=colors[ma], width=1.5), 
            name=ma), row=1, col=1)
    
 
    
    # 3. 畫成交量柱狀圖 (放在第二層)
    fig.add_trace(go.Bar(
    x=df['t'], y=df['v'],
    name='Volume', marker_color='gray'
            ), row=2, col=1)
    
       # 4. 美化圖表
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark', # 使用深色模式，看起來更專業
        annotations=[
            dict(
                text=f"Timeframe: {timeframe}",
                xref="paper", yref="paper",
                x=0.02, y=1.05,  # x=0 靠左，y=1.05 在圖表上方
                showarrow=False,
                font=dict(size=14, color="white"),
                align="left"
            )
        ]
        
    )
    
    
    
    return fig

# 使用方式：
# df_chart = plot_candlestick(response.json(), "AAPL")
# df_chart.show()


# ================Run=================
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


def extract_query_params(query: str) -> AlpacaQueryParams:
    """用 LLM 把一句話轉成 Alpaca 查詢參數（類別、代號、即時/歷史、日期、要不要畫圖）。
    獨立成函式，讓 evals/market/eval_market_params.py 可以只評估這一步，不用真的去打 Alpaca。"""
    checkprice_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7,api_key = os.environ["ANTHROPIC_API_KEY"].strip()).with_structured_output(AlpacaQueryParams)
    return checkprice_llm.invoke([
            ("user",query)
        ])


@tool
def market_price_check(query:str) -> dict:
    """
    查詢市場價格的工具。可以用於查詢美股、比特幣資產價格。
    當使用者詢問關於股票、資產或市場價格相關的問題時，使用此工具。
    此工具可以查詢一段時間價格：mode = history , 或是最新價格 mode = latest
    在mode = history的情況下，可以出蠟燭圖

    Returns:
        dict: 固定形狀 {"success": bool, "data": dict | None, "error": str | None}。
        data 成功時是 {"report": str, "fig": Figure | None, "symbols": str, "mode": str}，
        這個工具永遠不會把例外丟給呼叫端。
    """
    try:
        decision = extract_query_params(query)

        # output decision :
        # AlpacaQueryParams(category='crypto', symbols='BTC/USD', mode='latest', timeframe=None, start=None, end=None, feed=None, fig=False)

        result = fetch_alpaca_data(decision)

        # fetch_alpaca_data 失敗時回傳的是 {"error": ...}，跟成功時的 {"bars": ...} 不同形狀，
        # 這裡先攔截，避免 generate_market_report 拿到殘缺資料炸出難懂的 TypeError，
        # 也讓使用者看到 Alpaca 真正回傳的失敗原因。
        if "error" in result:
            return {"success": False, "data": None, "error": result["error"]}

        report = generate_market_report(result, decision.symbols, decision.mode)

        fig = None
        if decision.fig:
            fig = plot_candlestick(
                result,
                decision.symbols,
                decision.timeframe
            )

        return {
            "success": True,
            "data": {"report": report, "fig": fig, "symbols": decision.symbols, "mode": decision.mode},
            "error": None,
        }

    except Exception as e:
        error_msg = f"查詢市場價格時發生錯誤: {str(e)}"
        logger.error("[market_price_check] %s", error_msg)
        return {"success": False, "data": None, "error": error_msg}
    
    

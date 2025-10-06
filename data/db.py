import os
from pathlib import Path
import akshare as ak
import pandas as pd

def get_stock_data_ak(stock_code="000001",period="daily",start_date="20150101",end_date="20251001"):
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / f"{stock_code}_{period}_{start_date}_{end_date}.csv"

    df = pd.DataFrame()

    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            df.rename(
                columns={
                    "日期": "timestamps",
                    "股票代码": "code",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_chg",
                    "涨跌额": "change",
                    "换手率": "turnover",
                },
                inplace=True,
            )
            df.to_csv(file_path)
        except Exception as e:
            return None, e

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df, None

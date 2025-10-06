import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from data import db


def plotly_prediction(kline_df, pred_df):
    """
    绘制K线图及预测结果
    """
    # 创建子图：主图为K线，副图为成交量
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("价格走势", "成交量"),
        row_heights=[0.7, 0.3],
    )

    # 绘制实际K线图
    fig.add_trace(
        go.Candlestick(
            name="实际价格走势",
            x=kline_df.index,
            open=kline_df["open"],
            high=kline_df["high"],
            low=kline_df["low"],
            close=kline_df["close"],
        ),
        row=1,
        col=1,
    )
    # 绘制成交量
    fig.add_trace(
        go.Bar(
            x=kline_df.index,
            y=kline_df["volume"],
            name="成交量",
            marker_color=[
                "red" if kline_df["close"][i] < kline_df["open"][i] else "green"
                for i in range(len(kline_df))
            ],
        ),
        row=2,
        col=1,
    )

    # 绘制预测K线图（使用不同颜色）
    fig.add_trace(
        go.Candlestick(
            name="预测价格走势",
            x=pred_df.index,
            open=pred_df["open"],
            high=pred_df["high"],
            low=pred_df["low"],
            close=pred_df["close"],
            increasing_line_color="pink",
            decreasing_line_color="gray",
        ),
        row=1,
        col=1,
    )

    # 更新图表布局
    fig.update_layout(
        title="股票价格K线图及预测",
        xaxis_title="日期",
        yaxis_title="价格",
        template="plotly_white",
        height=1200,
    )

    # 隐藏非交易日期
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    fig.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
df, _ = db.get_stock_data_ak("600519", "daily", "20150101", "20251006")

x_len = 400  # 输入数据长度
p_len = 120  # 输出输出长度

# 获取最后x_len条数据
x_df = df.loc[-x_len:]
# 获取最后x_len条数据的时间戳
x_timestamp = x_df["timestamps"]
# 推导生成p_len个未来时间戳
y_timestamp = pd.Series(
    name="timestamps",
    data=pd.date_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(days=1),
        periods=p_len,
        freq="D",
    ),
)


# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=p_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(x_df.tail())


# visualize
df.set_index("timestamps", inplace=True)
plotly_prediction(df, pred_df)

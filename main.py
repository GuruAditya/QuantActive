
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Stock Strategy Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian Stock Strategy Analyzer")
st.caption(
    "Explore historical prices, technical indicators, "
    "and strategy backtests."
)


# -----------------------------
# STOCK UNIVERSE
# -----------------------------
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "Tata Motors": "TMPV.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
}


# -----------------------------
# DATA DOWNLOAD
# -----------------------------
@st.cache_data(ttl=3600)
def download_data(ticker, years):
    data = yf.download(
        ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.dropna(subset=["Close"])
    data.index.name = "Date"

    return data


# -----------------------------
# INDICATORS
# -----------------------------
def calculate_indicators(data, fast, slow, rsi_period, momentum_period):
    df = data.copy()

    df["SMA_Fast"] = df["Close"].rolling(fast).mean()
    df["SMA_Slow"] = df["Close"].rolling(slow).mean()

    # Wilder-style RSI smoothing
    change = df["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # If there are gains but no losses, RSI is 100.
    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "RSI"
    ] = 100

    # If there are neither gains nor losses, RSI is neutral.
    df.loc[
        (avg_loss == 0) & (avg_gain == 0), "RSI"
    ] = 50

    df["Momentum"] = (
        df["Close"] / df["Close"].shift(momentum_period) - 1
    )

    return df


# -----------------------------
# STRATEGY SIGNALS
# -----------------------------
def create_signal(df, strategy, rsi_buy, rsi_exit):
    if strategy == "Moving Average Crossover":
        signal = (
            df["SMA_Fast"] > df["SMA_Slow"]
        ).astype(int)

        valid = df["SMA_Slow"].notna()

    elif strategy == "RSI":
        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        for i, value in enumerate(df["RSI"]):
            if pd.isna(value):
                position = 0
            elif value < rsi_buy:
                position = 1
            elif value > rsi_exit:
                position = 0

            signal.iloc[i] = position

        valid = df["RSI"].notna()

    else:  # Momentum
        signal = (
            df["Momentum"] > 0
        ).astype(int)

        valid = df["Momentum"].notna()

    return signal.where(valid, 0)


# -----------------------------
# BACKTEST
# -----------------------------
def backtest(df, signal, cost_percent):
    result = df.copy()

    result["Market_Return"] = result["Close"].pct_change()

    # Avoid using today's closing signal for today's return.
    result["Position"] = signal.shift(1).fillna(0)

    turnover = result["Position"].diff().abs()
    turnover.iloc[0] = abs(result["Position"].iloc[0])

    cost = cost_percent / 100

    result["Strategy_Return"] = (
        result["Position"] * result["Market_Return"]
        - turnover * cost
    )

    result["Strategy_Return"] = (
        result["Strategy_Return"].fillna(0)
    )

    result["Equity"] = (
        1 + result["Strategy_Return"]
    ).cumprod()

    result["Buy_Hold"] = (
        1 + result["Market_Return"].fillna(0)
    ).cumprod()

    result["Drawdown"] = (
        result["Equity"] / result["Equity"].cummax() - 1
    )

    return result


def calculate_metrics(df):
    returns = df["Strategy_Return"]
    equity = df["Equity"]

    elapsed_years = (
        (df.index[-1] - df.index[0]).days / 365.25
    )

    total_return = equity.iloc[-1] - 1

    cagr = (
        equity.iloc[-1] ** (1 / elapsed_years) - 1
        if elapsed_years > 0 else np.nan
    )

    volatility = returns.std() * np.sqrt(252)

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 0 else np.nan
    )

    max_drawdown = df["Drawdown"].min()

    return total_return, cagr, volatility, sharpe, max_drawdown


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Configuration")

selected_stock = st.sidebar.selectbox(
    "Choose a stock",
    options=list(STOCKS.keys())
)

ticker = STOCKS[selected_stock]

years = st.sidebar.slider(
    "Historical period (years)",
    min_value=1,
    max_value=10,
    value=10
)

strategy = st.sidebar.selectbox(
    "Trading strategy",
    [
        "Moving Average Crossover",
        "RSI",
        "Momentum"
    ]
)

st.sidebar.subheader("Indicator parameters")

fast = st.sidebar.slider(
    "Fast moving average",
    5, 100, 20
)

slow = st.sidebar.slider(
    "Slow moving average",
    20, 250, 50
)

rsi_period = st.sidebar.slider(
    "RSI period",
    5, 30, 14
)

rsi_buy = st.sidebar.slider(
    "RSI entry threshold",
    10, 45, 30
)

rsi_exit = st.sidebar.slider(
    "RSI exit threshold",
    55, 90, 70
)

momentum_period = st.sidebar.slider(
    "Momentum lookback (trading days)",
    10, 120, 60
)

cost_percent = st.sidebar.number_input(
    "Estimated cost per one-way trade (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.1,
    step=0.05
)


# -----------------------------
# LOAD AND PROCESS DATA
# -----------------------------
if fast >= slow:
    st.sidebar.warning(
        "For a conventional crossover, fast MA should be "
        "shorter than slow MA."
    )

data = download_data(ticker, years)

if data.empty:
    st.error(
        "No data returned. Check the ticker or try again later."
    )
    st.stop()

df = calculate_indicators(
    data,
    fast,
    slow,
    rsi_period,
    momentum_period
)

signal = create_signal(
    df,
    strategy,
    rsi_buy,
    rsi_exit
)

result = backtest(df, signal, cost_percent)

st.subheader(f"{selected_stock} ({ticker})")

st.caption(
    f"Data from {result.index.min().date()} "
    f"to {result.index.max().date()} | "
    f"{len(result):,} trading sessions"
)


# -----------------------------
# PERFORMANCE METRICS
# -----------------------------
total, cagr, volatility, sharpe, drawdown = (
    calculate_metrics(result)
)

buy_hold_return = result["Buy_Hold"].iloc[-1] - 1

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Strategy Return", f"{total:.2%}")
c2.metric("CAGR", f"{cagr:.2%}")
c3.metric("Buy & Hold", f"{buy_hold_return:.2%}")
c4.metric("Sharpe Ratio", f"{sharpe:.2f}")
c5.metric("Max Drawdown", f"{drawdown:.2%}")


# -----------------------------
# PRICE AND INDICATOR CHART
# -----------------------------
price_fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=("Adjusted Historical Price", "RSI")
)

price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Adjusted Close"
    ),
    row=1,
    col=1
)

price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SMA_Fast"],
        name=f"SMA {fast}"
    ),
    row=1,
    col=1
)

price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SMA_Slow"],
        name=f"SMA {slow}"
    ),
    row=1,
    col=1
)

price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["RSI"],
        name="RSI",
        line=dict(width=1)
    ),
    row=2,
    col=1
)

price_fig.add_hline(
    y=70, line_dash="dash",
    row=2, col=1
)

price_fig.add_hline(
    y=30, line_dash="dash",
    row=2, col=1
)

price_fig.update_yaxes(
    title_text="Price (₹)", row=1, col=1
)

price_fig.update_yaxes(
    title_text="RSI", range=[0, 100], row=2, col=1
)

price_fig.update_layout(
    height=650,
    hovermode="x unified",
    legend=dict(orientation="h")
)

st.plotly_chart(price_fig, use_container_width=True)


# -----------------------------
# EQUITY CURVE
# -----------------------------
st.subheader("Strategy vs Buy & Hold")

equity_fig = go.Figure()

equity_fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["Equity"],
        name="Strategy"
    )
)

equity_fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["Buy_Hold"],
        name="Buy & Hold"
    )
)

equity_fig.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Growth of ₹1",
    hovermode="x unified"
)

st.plotly_chart(equity_fig, use_container_width=True)


# -----------------------------
# HISTORICAL DATA TABLE
# -----------------------------
with st.expander("View historical data and indicators"):
    st.dataframe(
        result.tail(100).sort_index(ascending=False),
        use_container_width=True
    )

st.download_button(
    "Download historical data and backtest CSV",
    data=result.to_csv(index=True).encode("utf-8"),
    file_name=f"{ticker.replace('.', '_')}_backtest.csv",
    mime="text/csv"
)

st.caption(
    "Educational backtest only. Historical performance does not "
    "predict future results. Adjusted prices, transaction costs, "
    "taxes, slippage and execution assumptions affect results."
)

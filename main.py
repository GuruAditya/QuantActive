import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_score,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(
    page_title="Stock Strategy Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian Stock Strategy Analyzer")
st.caption(
    "Backtest rule-based strategies with a cash-constrained portfolio, "
    "then stress-test them with optimization, walk-forward analysis, "
    "market-regime breakdowns and a simple ML model."
)

# -----------------------------
# CONSTANTS
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

MA_STRATEGY = "Moving Average Crossover"
RSI_STRATEGY = "RSI"
MOM_STRATEGY = "Momentum"
STRATEGIES = [MA_STRATEGY, RSI_STRATEGY, MOM_STRATEGY]

TRADE_COLS = [
    "Entry Date", "Exit Date", "Entry Price", "Exit Price", "Shares",
    "P&L (₹)", "Return", "Days Held", "Status"
]

PCT_KEYS = {
    "Total Return", "CAGR", "Volatility", "Max Drawdown",
    "Win Rate", "Avg Trade Return"
}


# =====================================================================
# DATA LOADING (Yahoo Finance or CSV upload)
# =====================================================================
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

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.dropna(subset=["Close"])
    data.index.name = "Date"
    return data


@st.cache_data
def parse_csv(file_bytes):
    """Parse an OHLCV CSV into the standard Open/High/Low/Close/Volume frame."""
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes), thousands=",")
    except Exception as exc:
        raise ValueError(f"Could not read the CSV file: {exc}")

    raw.columns = [str(c).strip() for c in raw.columns]
    lookup = {c.lower().replace("_", " "): c for c in raw.columns}

    def pick(*names):
        for name in names:
            if name in lookup:
                return lookup[name]
        return None

    date_col = pick("date", "datetime", "timestamp", "time")
    if date_col is None:
        date_col = raw.columns[0]

    close_col = pick("close", "adj close", "adjusted close", "close price")
    if close_col is None:
        raise ValueError(
            "The CSV needs a 'Close' column "
            f"(found: {', '.join(raw.columns)})."
        )

    try:
        dates = pd.to_datetime(raw[date_col], errors="coerce")
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
    except (AttributeError, TypeError, ValueError):
        dates = pd.to_datetime(
            raw[date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)

    def numeric(col):
        return pd.to_numeric(raw[col], errors="coerce") if col else None

    close = numeric(close_col)
    out = pd.DataFrame({"Close": close})

    for name in ("Open", "High", "Low"):
        col = pick(name.lower())
        series = numeric(col)
        out[name] = series if series is not None else close

    vol_col = pick("volume", "vol", "shares traded")
    vol = numeric(vol_col)
    out["Volume"] = vol if vol is not None else 0.0

    out.index = dates
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    out["Open"] = out["Open"].fillna(out["Close"])
    out["High"] = out["High"].fillna(out["Close"])
    out["Low"] = out["Low"].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(0)
    out.index.name = "Date"

    if len(out) < 60:
        raise ValueError(
            "Too few valid rows after cleaning "
            f"({len(out)}). Need at least 60 daily rows."
        )

    return out[["Open", "High", "Low", "Close", "Volume"]]


# =====================================================================
# INDICATORS AND SIGNALS
# =====================================================================
def rsi_series(close, period):
    """Wilder-style RSI."""
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.loc[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi.loc[(avg_loss == 0) & (avg_gain == 0)] = 50
    return rsi


def calculate_indicators(data, fast, slow, rsi_period, momentum_period):
    df = data.copy()
    df["SMA_Fast"] = df["Close"].rolling(fast).mean()
    df["SMA_Slow"] = df["Close"].rolling(slow).mean()
    df["RSI"] = rsi_series(df["Close"], rsi_period)
    df["Momentum"] = (
        df["Close"] / df["Close"].shift(momentum_period) - 1
    )
    return df


def build_signal(df, strategy, p):
    """Return a 0/1 numpy array: the desired position at each day's close."""
    close = df["Close"]

    if strategy == MA_STRATEGY:
        fast = close.rolling(p["fast"]).mean()
        slow = close.rolling(p["slow"]).mean()
        return ((fast > slow) & slow.notna()).astype(int).to_numpy()

    if strategy == RSI_STRATEGY:
        values = rsi_series(close, p["rsi_period"]).to_numpy()
        signal = np.zeros(len(values), dtype=int)
        position = 0
        for i, value in enumerate(values):
            if np.isnan(value):
                position = 0
            elif value < p["rsi_buy"]:
                position = 1
            elif value > p["rsi_exit"]:
                position = 0
            signal[i] = position
        return signal

    momentum = close / close.shift(p["mom_period"]) - 1
    return (momentum > 0).astype(int).to_numpy()


def to_target(signal):
    """Signal at close of day t is executed on day t+1 (no look-ahead)."""
    target = np.roll(np.asarray(signal, dtype=int), 1)
    target[0] = 0
    return target


# =====================================================================
# PORTFOLIO SIMULATION (cash constraint, whole shares, transaction costs)
# =====================================================================
def simulate(open_, close, target, capital, cost):
    """
    Long-only, all-in/all-out simulation.

    - Orders are filled at the day's open.
    - Only whole shares can be bought; leftover cash stays idle (0% return).
    - `cost` is charged on both the buy and the sell value.
    """
    n = len(close)
    equity = np.empty(n)
    cash = float(capital)
    shares = 0
    entry = None
    trades = []

    for i in range(n):
        price = open_[i]
        if not np.isfinite(price) or price <= 0:
            price = close[i]

        if target[i] == 1 and shares == 0:
            qty = int(cash // (price * (1 + cost)))
            if qty > 0:
                outlay = qty * price * (1 + cost)
                cash -= outlay
                shares = qty
                entry = (i, price, outlay)

        elif target[i] == 0 and shares > 0:
            proceeds = shares * price * (1 - cost)
            cash += proceeds
            trades.append((
                entry[0], i, entry[1], price, shares,
                proceeds - entry[2], proceeds / entry[2] - 1, True
            ))
            shares = 0
            entry = None

        equity[i] = cash + shares * close[i]

    if shares > 0:
        value = shares * close[-1]
        trades.append((
            entry[0], n - 1, entry[1], close[-1], shares,
            value - entry[2], value / entry[2] - 1, False
        ))

    return equity, trades


def run_backtest(df, target, capital, cost):
    equity_values, raw_trades = simulate(
        df["Open"].to_numpy(float),
        df["Close"].to_numpy(float),
        np.asarray(target),
        capital,
        cost
    )

    equity = pd.Series(equity_values, index=df.index, name="Equity")
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / capital - 1

    idx = df.index
    rows = []
    for ent, ext, p_in, p_out, qty, pnl, ret, closed in raw_trades:
        rows.append({
            "Entry Date": idx[ent],
            "Exit Date": idx[ext],
            "Entry Price": p_in,
            "Exit Price": p_out,
            "Shares": qty,
            "P&L (₹)": pnl,
            "Return": ret,
            "Days Held": (idx[ext] - idx[ent]).days,
            "Status": "Closed" if closed else "Open",
        })
    trades = pd.DataFrame(rows, columns=TRADE_COLS)

    peak = equity.cummax().clip(lower=capital)

    return {
        "equity": equity,
        "returns": returns,
        "drawdown": equity / peak - 1,
        "trades": trades,
        "position": pd.Series(np.asarray(target), index=idx),
        "exposure": float(np.mean(target)),
    }


def empty_trades():
    return pd.DataFrame(columns=TRADE_COLS)


def slice_trades(trades, start, end):
    if trades.empty:
        return trades
    mask = (trades["Exit Date"] >= start) & (trades["Exit Date"] <= end)
    return trades[mask]


# =====================================================================
# PERFORMANCE METRICS
# =====================================================================
def compute_metrics(returns, trades, rf=0.0):
    returns = returns.dropna()
    keys = [
        "Total Return", "CAGR", "Volatility", "Sharpe Ratio",
        "Max Drawdown", "Win Rate", "Trades",
        "Avg Trade Return", "Profit Factor"
    ]
    if len(returns) < 2:
        return {k: np.nan for k in keys}

    equity = (1 + returns).cumprod()
    total = equity.iloc[-1] - 1

    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years >= 1 and equity.iloc[-1] > 0 else np.nan
    )

    std = returns.std()
    volatility = std * np.sqrt(252)
    sharpe = (
        (returns.mean() - rf / 252) / std * np.sqrt(252)
        if std > 0 else np.nan
    )

    peak = np.maximum(equity.cummax(), 1.0)
    max_dd = (equity / peak - 1).min()

    closed = (
        trades[trades["Status"] == "Closed"] if len(trades) else trades
    )
    if len(closed):
        pnl = closed["P&L (₹)"].astype(float)
        win_rate = (pnl > 0).mean()
        gains = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        profit_factor = gains / losses if losses > 0 else np.nan
        avg_trade = closed["Return"].astype(float).mean()
    else:
        win_rate = profit_factor = avg_trade = np.nan

    return {
        "Total Return": total,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Trades": len(closed),
        "Avg Trade Return": avg_trade,
        "Profit Factor": profit_factor,
    }


def fmt_metric(key, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if key in PCT_KEYS:
        return f"{value:.2%}"
    if key == "Trades":
        return f"{int(value)}"
    return f"{value:.2f}"


def metrics_table(named_metrics):
    return pd.DataFrame({
        name: {k: fmt_metric(k, v) for k, v in m.items()}
        for name, m in named_metrics.items()
    })


def line_fig(series_dict, ytitle, height=400, title=None):
    fig = go.Figure()
    for name, series in series_dict.items():
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name))
    fig.update_layout(
        height=height,
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        hovermode="x unified",
        legend=dict(orientation="h")
    )
    return fig


# =====================================================================
# PARAMETER OPTIMIZATION
# =====================================================================
def param_grid(strategy):
    """Deliberately small grids to limit over-fitting."""
    if strategy == MA_STRATEGY:
        return [
            {"fast": f, "slow": s}
            for f in (10, 20, 30, 50)
            for s in (50, 100, 150, 200)
            if f < s
        ]
    if strategy == RSI_STRATEGY:
        return [
            {"rsi_period": p, "rsi_buy": b, "rsi_exit": e}
            for p in (10, 14, 21)
            for b in (25, 30, 35)
            for e in (60, 70, 75)
        ]
    return [{"mom_period": m} for m in (20, 40, 60, 90, 120)]


def precompute_targets(df, strategy):
    """Indicators are causal, so signals are built once on full history."""
    return [
        (p, to_target(build_signal(df, strategy, p)))
        for p in param_grid(strategy)
    ]


def quick_score(equity, capital, objective, rf):
    previous = np.concatenate(([capital], equity[:-1]))
    returns = equity / previous - 1
    total = equity[-1] / capital - 1

    if objective == "Total return":
        value = total
    elif objective == "Sharpe ratio":
        std = returns.std(ddof=1) if len(returns) > 1 else 0
        value = (
            (returns.mean() - rf / 252) / std * np.sqrt(252)
            if std > 0 else -np.inf
        )
    else:  # Return / max drawdown
        peak = np.maximum.accumulate(np.maximum(equity, capital))
        max_dd = (equity / peak - 1).min()
        value = total / abs(max_dd) if max_dd < 0 else (
            total if total > 0 else -np.inf
        )

    return value if np.isfinite(value) else -np.inf


def score_grid(targets, o, c, a, b, capital, cost, objective, rf):
    scores = []
    for _, target in targets:
        equity, _ = simulate(
            o[a:b], c[a:b], target[a:b], capital, cost
        )
        scores.append(quick_score(equity, capital, objective, rf))
    return np.array(scores)


def params_to_text(p):
    return ", ".join(f"{k}={v}" for k, v in p.items())


@st.cache_data(show_spinner="Optimizing parameters on training data...")
def run_optimization(df, strategy, train_frac, capital, cost, rf, objective):
    targets = precompute_targets(df, strategy)
    o = df["Open"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    n = len(df)
    split = int(n * train_frac)

    train_scores = score_grid(
        targets, o, c, 0, split, capital, cost, objective, rf
    )
    test_scores = score_grid(
        targets, o, c, split, n, capital, cost, objective, rf
    )
    best = int(np.argmax(train_scores))

    table = pd.DataFrame([p for p, _ in targets])
    table["Train score"] = train_scores
    table["Test score"] = test_scores
    table = table.replace(-np.inf, np.nan).sort_values(
        "Train score", ascending=False
    )

    best_params, best_target = targets[best]
    return {
        "best_params": best_params,
        "table": table,
        "split": split,
        "result": run_backtest(df, best_target, capital, cost),
        "buy_hold": run_backtest(df, np.ones(n), capital, cost),
    }


# =====================================================================
# WALK-FORWARD EVALUATION
# =====================================================================
@st.cache_data(show_spinner="Running walk-forward analysis...")
def run_walk_forward(
    df, strategy, train_days, test_days, anchored,
    capital, cost, rf, objective
):
    targets = precompute_targets(df, strategy)
    o = df["Open"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    n = len(df)

    rows, oos_returns, oos_trades = [], [], []
    first_test = None
    pos = train_days

    while pos + test_days <= n:
        start = 0 if anchored else pos - train_days
        scores = score_grid(
            targets, o, c, start, pos, capital, cost, objective, rf
        )
        k = int(np.argmax(scores))
        params, target = targets[k]

        train_df = df.iloc[start:pos]
        test_df = df.iloc[pos:pos + test_days]

        train_res = run_backtest(train_df, target[start:pos], capital, cost)
        test_res = run_backtest(
            test_df, target[pos:pos + test_days], capital, cost
        )
        bh_res = run_backtest(test_df, np.ones(test_days), capital, cost)

        m_is = compute_metrics(
            train_res["returns"], train_res["trades"], rf
        )
        m_oos = compute_metrics(
            test_res["returns"], test_res["trades"], rf
        )
        m_bh = compute_metrics(bh_res["returns"], empty_trades(), rf)

        rows.append({
            "Fold": len(rows) + 1,
            "Train Start": train_df.index[0].date(),
            "Train End": train_df.index[-1].date(),
            "Test Start": test_df.index[0].date(),
            "Test End": test_df.index[-1].date(),
            "Best Parameters": params_to_text(params),
            "IS Sharpe": m_is["Sharpe Ratio"],
            "OOS Sharpe": m_oos["Sharpe Ratio"],
            "IS Return": m_is["Total Return"],
            "OOS Return": m_oos["Total Return"],
            "B&H OOS Return": m_bh["Total Return"],
            "OOS Trades": m_oos["Trades"],
        })

        if first_test is None:
            first_test = pos
        oos_returns.append(test_res["returns"])
        oos_trades.append(test_res["trades"])
        pos += test_days

    if not rows:
        return None

    last_end = first_test + test_days * len(rows)
    bh_span = run_backtest(
        df.iloc[first_test:last_end],
        np.ones(last_end - first_test),
        capital,
        cost
    )

    trades = pd.concat(oos_trades, ignore_index=True)
    return {
        "folds": pd.DataFrame(rows),
        "oos_returns": pd.concat(oos_returns),
        "oos_trades": trades,
        "bh_returns": bh_span["returns"],
    }


# =====================================================================
# MARKET CONDITIONS
# =====================================================================
def classify_regimes(close):
    sma200 = close.rolling(200).mean()
    trend = np.where(
        close > sma200 * 1.02, "Bull",
        np.where(close < sma200 * 0.98, "Bear", "Sideways")
    )
    trend = pd.Series(trend, index=close.index).where(sma200.notna())

    vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    median = vol.expanding(min_periods=60).median()
    volatility = pd.Series(
        np.where(vol > median, "High volatility", "Low volatility"),
        index=close.index
    ).where(vol.notna() & median.notna())

    return trend, volatility


def regime_table(regime, strategy_returns, bh_returns, order):
    rows = []
    classified = regime.notna().sum()
    for name in order:
        mask = regime == name
        if mask.sum() < 5:
            continue
        s = strategy_returns[mask]
        b = bh_returns[mask]
        rows.append({
            "Regime": name,
            "Days": int(mask.sum()),
            "% of Time": mask.sum() / classified,
            "Strategy Ann. Return": s.mean() * 252,
            "Buy & Hold Ann. Return": b.mean() * 252,
            "Strategy Sharpe": (
                s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan
            ),
            "Buy & Hold Sharpe": (
                b.mean() / b.std() * np.sqrt(252) if b.std() > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def show_regime_section(title, regime, res, bh, order):
    st.markdown(f"**{title}**")
    table = regime_table(
        regime.shift(1), res["returns"], bh["returns"], order
    )
    if table.empty:
        st.info(
            "Not enough history to classify regimes "
            "(roughly 200+ trading days are needed)."
        )
        return

    display = table.copy()
    display["% of Time"] = display["% of Time"].map(lambda v: f"{v:.1%}")
    for col in ("Strategy Ann. Return", "Buy & Hold Ann. Return"):
        display[col] = display[col].map(lambda v: f"{v:.2%}")
    for col in ("Strategy Sharpe", "Buy & Hold Sharpe"):
        display[col] = display[col].map(
            lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
        )
    st.dataframe(display, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=table["Regime"], y=table["Strategy Ann. Return"],
        name="Strategy"
    ))
    fig.add_trace(go.Bar(
        x=table["Regime"], y=table["Buy & Hold Ann. Return"],
        name="Buy & Hold"
    ))
    fig.update_layout(
        barmode="group", height=320,
        yaxis_title="Annualized return (avg daily × 252)",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# MACHINE LEARNING (price direction)
# =====================================================================
def make_features(df):
    close = df["Close"]
    volume = df["Volume"].replace(0, np.nan)
    ret = close.pct_change()

    X = pd.DataFrame(index=df.index)
    X["Return 1d"] = ret
    X["Return 5d"] = close.pct_change(5)
    X["Return 10d"] = close.pct_change(10)
    X["Return 20d"] = close.pct_change(20)
    X["RSI 14"] = rsi_series(close, 14)
    X["Price vs SMA20"] = close / close.rolling(20).mean() - 1
    X["SMA20 vs SMA50"] = (
        close.rolling(20).mean() / close.rolling(50).mean() - 1
    )
    X["Volatility 20d"] = ret.rolling(20).std()
    X["Momentum 60d"] = close.pct_change(60)
    X["Volume vs 20d avg"] = (
        volume / volume.rolling(20).mean() - 1
    ).fillna(0)
    return X.replace([np.inf, -np.inf], np.nan)


@st.cache_data(show_spinner="Training model...")
def run_ml(
    df, model_name, horizon, train_frac, threshold, capital, cost, rf
):
    X = make_features(df)
    forward = df["Close"].shift(-horizon) / df["Close"] - 1
    y = (forward > 0).astype(float).where(forward.notna())

    feats = X.dropna()
    labeled = feats.index.intersection(y.dropna().index)
    if len(labeled) < 300:
        return None

    split_pos = int(len(labeled) * train_frac)
    split_date = labeled[split_pos]
    # Purge the last `horizon` training rows: their labels look into the test set.
    train_idx = labeled[:max(split_pos - horizon, 0)]
    test_idx = labeled[split_pos:]

    y_train = y.loc[train_idx].astype(int)
    y_test = y.loc[test_idx].astype(int)
    if y_train.nunique() < 2 or len(test_idx) < 30:
        return None

    if model_name == "Logistic Regression":
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=30,
            random_state=42, n_jobs=-1
        )
    model.fit(X.loc[train_idx], y_train)

    test_all = feats.loc[split_date:]
    prob_all = pd.Series(
        model.predict_proba(test_all)[:, 1], index=test_all.index
    )

    train_pred = model.predict(X.loc[train_idx])
    test_prob = prob_all.loc[test_idx]
    test_pred = (test_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, test_prob)
    except ValueError:
        auc = np.nan

    if model_name == "Logistic Regression":
        importance = np.abs(model[-1].coef_[0])
    else:
        importance = model.feature_importances_
    importance = pd.Series(
        importance / importance.sum(), index=X.columns
    ).sort_values(ascending=False)

    segment = df.loc[test_all.index]
    signal = (prob_all >= threshold).astype(int).to_numpy()
    strategy_res = run_backtest(segment, to_target(signal), capital, cost)
    bh_res = run_backtest(segment, np.ones(len(segment)), capital, cost)

    return {
        "split_date": split_date,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_acc": accuracy_score(y_train, train_pred),
        "test_acc": accuracy_score(y_test, test_pred),
        "baseline_acc": y_test.mean(),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "auc": auc,
        "confusion": pd.DataFrame(
            confusion_matrix(y_test, test_pred, labels=[0, 1]),
            index=["Actual Down", "Actual Up"],
            columns=["Predicted Down", "Predicted Up"]
        ),
        "importance": importance,
        "prob": prob_all,
        "strategy": strategy_res,
        "buy_hold": bh_res,
        "metrics": compute_metrics(
            strategy_res["returns"], strategy_res["trades"], rf
        ),
        "bh_metrics": compute_metrics(
            bh_res["returns"], empty_trades(), rf
        ),
    }


# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.header("⚙️ Configuration")

source = st.sidebar.radio("Data source", ["Yahoo Finance", "Upload CSV"])
uploaded = None
selected_stock = None

if source == "Yahoo Finance":
    selected_stock = st.sidebar.selectbox(
        "Choose a stock", options=list(STOCKS.keys())
    )
else:
    uploaded = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])
    st.sidebar.caption(
        "Needs a date column and a Close column. Open, High, Low and "
        "Volume are optional (Open defaults to Close)."
    )

years = st.sidebar.slider("Historical period (years)", 1, 10, 10)

strategy = st.sidebar.selectbox("Trading strategy", STRATEGIES)

st.sidebar.subheader("Indicator parameters")
fast = st.sidebar.slider("Fast moving average", 5, 100, 20)
slow = st.sidebar.slider("Slow moving average", 20, 250, 50)
rsi_period = st.sidebar.slider("RSI period", 5, 30, 14)
rsi_buy = st.sidebar.slider("RSI entry threshold", 10, 45, 30)
rsi_exit = st.sidebar.slider("RSI exit threshold", 55, 90, 70)
momentum_period = st.sidebar.slider(
    "Momentum lookback (trading days)", 10, 120, 60
)

st.sidebar.subheader("Portfolio & costs")
capital = st.sidebar.number_input(
    "Initial capital (₹)",
    min_value=1000.0, max_value=1e9, value=100000.0, step=10000.0
)
cost_percent = st.sidebar.number_input(
    "Estimated cost per one-way trade (%)",
    min_value=0.0, max_value=2.0, value=0.1, step=0.05
)
risk_free = st.sidebar.number_input(
    "Risk-free rate for Sharpe (% per year)",
    min_value=0.0, max_value=15.0, value=0.0, step=0.5
) / 100

st.sidebar.subheader("Optimization")
objective = st.sidebar.selectbox(
    "Optimization objective",
    ["Sharpe ratio", "Total return", "Return / max drawdown"]
)

if fast >= slow:
    st.sidebar.warning(
        "For a conventional crossover, fast MA should be "
        "shorter than slow MA."
    )

cost = cost_percent / 100


# =====================================================================
# LOAD DATA
# =====================================================================
if source == "Yahoo Finance":
    ticker = STOCKS[selected_stock]
    data = download_data(ticker, years)
    title = f"{selected_stock} ({ticker})"
else:
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to begin.")
        st.stop()
    try:
        data = parse_csv(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    cutoff = data.index.max() - pd.DateOffset(years=years)
    data = data[data.index >= cutoff]
    ticker = uploaded.name.rsplit(".", 1)[0]
    title = f"{ticker} (uploaded CSV)"

if data.empty:
    st.error("No data returned. Check the ticker/file or try again later.")
    st.stop()

if len(data) < 100:
    st.error("Not enough rows to run a backtest (need at least 100).")
    st.stop()

params = {
    "fast": fast,
    "slow": slow,
    "rsi_period": rsi_period,
    "rsi_buy": rsi_buy,
    "rsi_exit": rsi_exit,
    "mom_period": momentum_period,
}

indicators = calculate_indicators(
    data, fast, slow, rsi_period, momentum_period
)
signal = build_signal(data, strategy, params)
res = run_backtest(data, to_target(signal), capital, cost)
bh = run_backtest(data, np.ones(len(data)), capital, cost)

metrics = compute_metrics(res["returns"], res["trades"], risk_free)
bh_metrics = compute_metrics(bh["returns"], empty_trades(), risk_free)

st.subheader(title)
st.caption(
    f"Data from {data.index.min().date()} to {data.index.max().date()} | "
    f"{len(data):,} trading sessions | Orders fill at the next day's open"
)

tab_overview, tab_compare, tab_opt, tab_wf, tab_regime, tab_ml = st.tabs([
    "📊 Backtest",
    "⚖️ Strategy Comparison",
    "🎯 Optimization (In vs Out-of-Sample)",
    "🔁 Walk-Forward",
    "🌦️ Market Conditions",
    "🤖 ML Model",
])


# =====================================================================
# TAB 1: BACKTEST
# =====================================================================
with tab_overview:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Strategy Return", fmt_metric("Total Return", metrics["Total Return"]))
    c2.metric("CAGR", fmt_metric("CAGR", metrics["CAGR"]))
    c3.metric("Buy & Hold", fmt_metric("Total Return", bh_metrics["Total Return"]))
    c4.metric("Sharpe Ratio", fmt_metric("Sharpe Ratio", metrics["Sharpe Ratio"]))
    c5.metric("Max Drawdown", fmt_metric("Max Drawdown", metrics["Max Drawdown"]))
    c6.metric("Win Rate", fmt_metric("Win Rate", metrics["Win Rate"]))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Final Portfolio Value", f"₹{res['equity'].iloc[-1]:,.0f}")
    d2.metric("Buy & Hold Value", f"₹{bh['equity'].iloc[-1]:,.0f}")
    d3.metric("Closed Trades", fmt_metric("Trades", metrics["Trades"]))
    d4.metric("Time in Market", f"{res['exposure']:.1%}")

    st.caption(
        "Latest signal: "
        + ("🟢 hold / buy" if signal[-1] == 1 else "⚪ stay in cash")
        + " (applies from the next session's open)."
    )

    # ----- Price, volume, RSI chart with entry/exit markers -----
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.55, 0.15, 0.30],
        subplot_titles=("Price with entry/exit signals", "Volume", "RSI")
    )
    fig.add_trace(go.Scatter(
        x=indicators.index, y=indicators["Close"], name="Close"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=indicators.index, y=indicators["SMA_Fast"], name=f"SMA {fast}"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=indicators.index, y=indicators["SMA_Slow"], name=f"SMA {slow}"
    ), row=1, col=1)

    trades = res["trades"]
    if len(trades):
        fig.add_trace(go.Scatter(
            x=trades["Entry Date"], y=trades["Entry Price"],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=10, color="green")
        ), row=1, col=1)
        closed = trades[trades["Status"] == "Closed"]
        fig.add_trace(go.Scatter(
            x=closed["Exit Date"], y=closed["Exit Price"],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=10, color="red")
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=indicators.index, y=indicators["Volume"], name="Volume",
        marker_color="gray", showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=indicators.index, y=indicators["RSI"], name="RSI",
        line=dict(width=1)
    ), row=3, col=1)
    fig.add_hline(y=rsi_exit, line_dash="dash", row=3, col=1)
    fig.add_hline(y=rsi_buy, line_dash="dash", row=3, col=1)

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Shares", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_layout(
        height=800, hovermode="x unified", legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----- Equity and drawdown -----
    st.subheader("Strategy vs Buy & Hold")
    st.plotly_chart(
        line_fig(
            {"Strategy": res["equity"], "Buy & Hold": bh["equity"]},
            "Portfolio value (₹)"
        ),
        use_container_width=True
    )
    st.plotly_chart(
        line_fig(
            {"Strategy": res["drawdown"], "Buy & Hold": bh["drawdown"]},
            "Drawdown", height=300, title="Drawdown"
        ),
        use_container_width=True
    )

    st.subheader("Performance summary")
    st.dataframe(
        metrics_table({"Strategy": metrics, "Buy & Hold": bh_metrics}),
        use_container_width=True
    )

    # ----- Trade log -----
    with st.expander("Trade log"):
        if trades.empty:
            st.info("The strategy made no trades in this period.")
        else:
            log = trades.copy()
            log["Entry Date"] = log["Entry Date"].dt.date
            log["Exit Date"] = log["Exit Date"].dt.date
            log["Return (%)"] = (log["Return"] * 100).round(2)
            log = log.drop(columns="Return")
            for col in ("Entry Price", "Exit Price", "P&L (₹)"):
                log[col] = log[col].round(2)
            st.dataframe(log, use_container_width=True, hide_index=True)
            st.download_button(
                "Download trade log (CSV)",
                data=log.to_csv(index=False).encode("utf-8"),
                file_name="trade_log.csv",
                mime="text/csv"
            )

    # ----- Data export -----
    export = indicators.copy()
    export["Signal"] = signal
    export["Position"] = res["position"]
    export["Strategy_Equity"] = res["equity"]
    export["Strategy_Return"] = res["returns"]
    export["Drawdown"] = res["drawdown"]
    export["Buy_Hold_Equity"] = bh["equity"]

    with st.expander("View historical data and indicators"):
        st.dataframe(
            export.tail(100).sort_index(ascending=False),
            use_container_width=True
        )

    st.download_button(
        "Download historical data and backtest CSV",
        data=export.to_csv(index=True).encode("utf-8"),
        file_name=f"{ticker.replace('.', '_')}_backtest.csv",
        mime="text/csv"
    )


# =====================================================================
# TAB 2: STRATEGY COMPARISON
# =====================================================================
with tab_compare:
    st.markdown(
        "All three strategies run with the current sidebar parameters "
        "and the same capital and cost assumptions."
    )

    compare_results = {}
    for name in STRATEGIES:
        sig = build_signal(data, name, params)
        compare_results[name] = run_backtest(
            data, to_target(sig), capital, cost
        )
    compare_results["Buy & Hold"] = bh

    compare_metrics = {
        name: compute_metrics(
            r["returns"],
            r["trades"] if name != "Buy & Hold" else empty_trades(),
            risk_free
        )
        for name, r in compare_results.items()
    }

    st.dataframe(metrics_table(compare_metrics), use_container_width=True)

    st.plotly_chart(
        line_fig(
            {n: r["equity"] for n, r in compare_results.items()},
            "Portfolio value (₹)", height=450,
            title="Equity curves"
        ),
        use_container_width=True
    )
    st.plotly_chart(
        line_fig(
            {n: r["drawdown"] for n, r in compare_results.items()},
            "Drawdown", height=320, title="Drawdowns"
        ),
        use_container_width=True
    )

    bar = go.Figure(go.Bar(
        x=list(compare_metrics.keys()),
        y=[m["Sharpe Ratio"] for m in compare_metrics.values()]
    ))
    bar.update_layout(
        height=300, title="Sharpe ratio", yaxis_title="Sharpe"
    )
    st.plotly_chart(bar, use_container_width=True)


# =====================================================================
# TAB 3: OPTIMIZATION (IN-SAMPLE VS OUT-OF-SAMPLE)
# =====================================================================
with tab_opt:
    st.markdown(
        "Parameters are chosen from a **small grid** using only the "
        "training portion of history, then evaluated on the unseen "
        "out-of-sample portion with those parameters held fixed."
    )

    train_pct = st.slider(
        "Training share of history (%)", 50, 90, 70, 5, key="opt_train"
    )

    opt = run_optimization(
        data, strategy, train_pct / 100, capital, cost, risk_free, objective
    )

    split = opt["split"]
    split_date = data.index[split]
    best = opt["best_params"]

    st.success(
        f"Best parameters on training data ({objective}): "
        f"**{params_to_text(best)}**"
    )
    st.caption(
        f"Training: {data.index[0].date()} → {data.index[split - 1].date()} "
        f"({split:,} sessions) | Out-of-sample: {split_date.date()} → "
        f"{data.index[-1].date()} ({len(data) - split:,} sessions)"
    )

    r = opt["result"]
    b = opt["buy_hold"]
    train_end = data.index[split - 1]

    def seg_metrics(result, start, end, with_trades=True):
        trades_part = (
            slice_trades(result["trades"], start, end)
            if with_trades else empty_trades()
        )
        return compute_metrics(
            result["returns"].loc[start:end], trades_part, risk_free
        )

    comparison = {
        "Strategy – In-sample": seg_metrics(r, data.index[0], train_end),
        "Strategy – Out-of-sample": seg_metrics(r, split_date, data.index[-1]),
        "Buy & Hold – In-sample": seg_metrics(
            b, data.index[0], train_end, False
        ),
        "Buy & Hold – Out-of-sample": seg_metrics(
            b, split_date, data.index[-1], False
        ),
    }
    st.dataframe(metrics_table(comparison), use_container_width=True)

    is_sharpe = comparison["Strategy – In-sample"]["Sharpe Ratio"]
    oos_sharpe = comparison["Strategy – Out-of-sample"]["Sharpe Ratio"]
    if pd.notna(is_sharpe) and pd.notna(oos_sharpe):
        if is_sharpe > 0 and oos_sharpe < 0.5 * is_sharpe:
            st.warning(
                f"Out-of-sample Sharpe ({oos_sharpe:.2f}) is much lower than "
                f"in-sample ({is_sharpe:.2f}). This is a typical sign of "
                "over-fitting."
            )
        else:
            st.info(
                f"In-sample Sharpe {is_sharpe:.2f} vs out-of-sample "
                f"{oos_sharpe:.2f}."
            )

    fig = line_fig(
        {"Strategy (best params)": r["equity"], "Buy & Hold": b["equity"]},
        "Portfolio value (₹)", height=420,
        title="Equity with the training-optimized parameters"
    )
    fig.add_vrect(
        x0=split_date, x1=data.index[-1],
        fillcolor="orange", opacity=0.12, line_width=0,
        annotation_text="Out-of-sample", annotation_position="top left"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Top 10 parameter sets by training score ({objective})**")
    top = opt["table"].head(10).copy()
    top["Train score"] = top["Train score"].round(3)
    top["Test score"] = top["Test score"].round(3)
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.caption(
        "Test scores are shown for diagnosis only. Selecting parameters by "
        "looking at them would defeat the out-of-sample test."
    )


# =====================================================================
# TAB 4: WALK-FORWARD
# =====================================================================
with tab_wf:
    st.markdown(
        "For each fold the best parameters are found on the training "
        "window, then traded on the following unseen test window. "
        "Test windows are stitched together into one out-of-sample record."
    )

    w1, w2, w3 = st.columns(3)
    mode = w1.radio(
        "Window type", ["Rolling window", "Anchored (expanding)"],
        key="wf_mode"
    )
    train_years = w2.slider("Training window (years)", 1, 5, 3, key="wf_train")
    test_months = w3.slider(
        "Test window (months)", 3, 12, 6, step=3, key="wf_test"
    )

    wf = run_walk_forward(
        data, strategy, train_years * 252, test_months * 21,
        mode.startswith("Anchored"), capital, cost, risk_free, objective
    )

    if wf is None:
        st.warning(
            "Not enough data for even one fold. Increase the historical "
            "period or shorten the training/test windows."
        )
    else:
        folds = wf["folds"]
        oos_m = compute_metrics(
            wf["oos_returns"], wf["oos_trades"], risk_free
        )
        bh_m = compute_metrics(wf["bh_returns"], empty_trades(), risk_free)

        st.markdown(f"**Stitched out-of-sample result ({len(folds)} folds)**")
        st.dataframe(
            metrics_table({
                "Walk-forward strategy (OOS)": oos_m,
                "Buy & Hold (same period)": bh_m,
            }),
            use_container_width=True
        )

        oos_equity = capital * (1 + wf["oos_returns"]).cumprod()
        bh_equity = capital * (1 + wf["bh_returns"]).cumprod()
        st.plotly_chart(
            line_fig(
                {
                    "Walk-forward strategy (OOS)": oos_equity,
                    "Buy & Hold": bh_equity,
                },
                "Portfolio value (₹)", height=400,
                title="Out-of-sample equity"
            ),
            use_container_width=True
        )

        mean_is = folds["IS Sharpe"].mean()
        mean_oos = folds["OOS Sharpe"].mean()
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg in-sample Sharpe", f"{mean_is:.2f}")
        k2.metric("Avg out-of-sample Sharpe", f"{mean_oos:.2f}")
        k3.metric(
            "OOS / IS Sharpe",
            f"{mean_oos / mean_is:.2f}" if mean_is and mean_is > 0 else "n/a"
        )

        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=folds["Fold"], y=folds["IS Sharpe"], name="In-sample"
        ))
        bar.add_trace(go.Bar(
            x=folds["Fold"], y=folds["OOS Sharpe"], name="Out-of-sample"
        ))
        bar.update_layout(
            barmode="group", height=320, xaxis_title="Fold",
            yaxis_title="Sharpe", title="In-sample vs out-of-sample Sharpe",
            legend=dict(orientation="h")
        )
        st.plotly_chart(bar, use_container_width=True)

        display = folds.copy()
        for col in ("IS Return", "OOS Return", "B&H OOS Return"):
            display[col] = display[col].map(lambda v: f"{v:.2%}")
        for col in ("IS Sharpe", "OOS Sharpe"):
            display[col] = display[col].map(
                lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
            )
        st.markdown("**Fold details and chosen parameters**")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Each fold starts fresh in cash with the initial capital. "
            "Parameters that change a lot between folds suggest an "
            "unstable strategy."
        )


# =====================================================================
# TAB 5: MARKET CONDITIONS
# =====================================================================
with tab_regime:
    st.markdown(
        "Daily strategy returns are grouped by the market regime known at "
        "the **previous close**, so the split uses no future information."
    )
    if years < 2:
        st.warning(
            "Regimes use a 200-day average, so use at least 2 years of "
            "history for a meaningful split."
        )

    trend, vol_regime = classify_regimes(data["Close"])

    show_regime_section(
        "Trend regime: Bull (price > 2% above 200-day SMA), "
        "Bear (> 2% below), Sideways (within ±2%)",
        trend, res, bh, ["Bull", "Sideways", "Bear"]
    )

    show_regime_section(
        "Volatility regime: 20-day volatility vs its running median",
        vol_regime, res, bh, ["Low volatility", "High volatility"]
    )

    price_fig = go.Figure()
    colors = {"Bull": "green", "Sideways": "gray", "Bear": "red"}
    for name in ("Bull", "Sideways", "Bear"):
        price_fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"].where(trend == name),
            name=name,
            mode="lines",
            line=dict(color=colors[name], width=1.5),
            connectgaps=False
        ))
    price_fig.update_layout(
        height=380, title="Price coloured by trend regime",
        yaxis_title="Price (₹)", hovermode="x unified",
        legend=dict(orientation="h")
    )
    st.plotly_chart(price_fig, use_container_width=True)


# =====================================================================
# TAB 6: ML MODEL
# =====================================================================
with tab_ml:
    st.markdown(
        "A simple classifier predicts whether the price will be higher "
        "`N` trading days from now, using lagged returns, RSI, "
        "moving-average ratios, volatility, momentum and volume. "
        "It is trained on the earlier part of history only and "
        "evaluated on the later part."
    )

    if not ML_AVAILABLE:
        st.warning(
            "scikit-learn is not installed. Run `pip install scikit-learn` "
            "and restart the app to enable this tab."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        model_name = m1.selectbox(
            "Model", ["Logistic Regression", "Random Forest"], key="ml_model"
        )
        horizon = m2.selectbox(
            "Prediction horizon (days)", [1, 5, 10], index=1, key="ml_h"
        )
        ml_train_pct = m3.slider(
            "Training share (%)", 50, 90, 70, 5, key="ml_split"
        )
        threshold = m4.slider(
            "Buy when P(up) ≥", 0.40, 0.65, 0.50, 0.01, key="ml_thr"
        )

        ml = run_ml(
            data, model_name, horizon, ml_train_pct / 100, threshold,
            capital, cost, risk_free
        )

        if ml is None:
            st.warning(
                "Not enough data (or only one class in the training set). "
                "Use a longer history."
            )
        else:
            st.caption(
                f"Train rows: {ml['n_train']:,} | Test rows: {ml['n_test']:,} "
                f"| Test period starts {ml['split_date'].date()}"
            )

            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Train accuracy", f"{ml['train_acc']:.1%}")
            a2.metric("Test accuracy", f"{ml['test_acc']:.1%}")
            a3.metric("Always-up baseline", f"{ml['baseline_acc']:.1%}")
            a4.metric("Test precision (up)", f"{ml['precision']:.1%}")
            a5.metric(
                "Test AUC",
                "n/a" if pd.isna(ml["auc"]) else f"{ml['auc']:.3f}"
            )

            if ml["test_acc"] <= ml["baseline_acc"] + 0.01:
                st.warning(
                    "Test accuracy is not meaningfully better than simply "
                    "guessing 'up' every time, so the model shows no "
                    "reliable predictive edge here. This is common with "
                    "daily price data."
                )

            left, right = st.columns(2)
            with left:
                st.markdown("**Confusion matrix (test set)**")
                st.dataframe(ml["confusion"], use_container_width=True)
            with right:
                st.markdown("**Relative feature importance**")
                imp = ml["importance"].sort_values()
                imp_fig = go.Figure(go.Bar(
                    x=imp.values, y=imp.index, orientation="h"
                ))
                imp_fig.update_layout(height=300, margin=dict(t=10))
                st.plotly_chart(imp_fig, use_container_width=True)

            st.markdown("**ML strategy vs Buy & Hold (out-of-sample only)**")
            st.dataframe(
                metrics_table({
                    "ML strategy": ml["metrics"],
                    "Buy & Hold": ml["bh_metrics"],
                }),
                use_container_width=True
            )
            st.plotly_chart(
                line_fig(
                    {
                        "ML strategy": ml["strategy"]["equity"],
                        "Buy & Hold": ml["buy_hold"]["equity"],
                    },
                    "Portfolio value (₹)", height=380
                ),
                use_container_width=True
            )

            prob_fig = line_fig(
                {"P(up)": ml["prob"]}, "Probability", height=260,
                title="Predicted probability of a price rise"
            )
            prob_fig.add_hline(y=threshold, line_dash="dash")
            st.plotly_chart(prob_fig, use_container_width=True)


st.caption(
    "Educational backtest only. Historical performance does not "
    "predict future results. Prices, transaction costs, taxes, slippage "
    "and execution assumptions (fills at next open, whole shares, idle cash "
    "earning 0%) all affect results."
)
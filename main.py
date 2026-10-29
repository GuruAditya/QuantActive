import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(
    page_title="QuantActive - Indian Stock Strategy Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 QuantActive - Strategy Backtesting & Stress-Testing Engine")
st.markdown(
    "A quantitative analysis platform featuring **execution constraints**, **out-of-sample walk-forward analysis**, "
    "**market regime splitting**, and **ML-driven direction classification**."
)

@st.cache_data
def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily stock data via yfinance and ensure standardized column formatting."""
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.dropna().copy()
    return df

def calculate_technical_indicators(df: pd.DataFrame, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.DataFrame:
    """Calculate core strategy indicators using strict causality."""
    data = df.copy()
    
    data['SMA_Fast'] = data['Close'].rolling(window=fast_ma).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow_ma).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['Daily_Return'] = data['Close'].pct_change()
    data['Vol_20'] = data['Daily_Return'].rolling(20).std() * np.sqrt(252)
    
    return data

def run_backtest(data: pd.DataFrame, initial_cash: float, fee_pct: float, strategy_type: str, **kwargs) -> pd.DataFrame:
    """
    Executes trades strictly at $t+1$ Open price based on signals generated at $t$ Close.
    Includes integer share allocation and double-sided transaction fees.
    """
    df = data.copy()
    n = len(df)
    
    if strategy_type == "MA Crossover":
        raw_signal = np.where(df['SMA_Fast'] > df['SMA_Slow'], 1, 0)
    elif strategy_type == "RSI Mean Reversion":
        buy_thresh = kwargs.get('rsi_buy', 30)
        exit_thresh = kwargs.get('rsi_exit', 70)
        
        sig = np.zeros(n)
        pos = 0
        rsi_vals = df['RSI'].values
        for i in range(n):
            if np.isnan(rsi_vals[i]):
                sig[i] = 0
                continue
            if pos == 0 and rsi_vals[i] < buy_thresh:
                pos = 1
            elif pos == 1 and rsi_vals[i] > exit_thresh:
                pos = 0
            sig[i] = pos
        raw_signal = sig
    else: # Buy & Hold
        raw_signal = np.ones(n)
        
    target_pos = np.roll(raw_signal, 1)
    target_pos[0] = 0  # No trade on Day 0
    
    cash = initial_cash
    shares = 0
    
    portfolio_values = []
    cash_history = []
    shares_history = []
    
    opens = df['Open'].values
    closes = df['Close'].values
    
    for i in range(n):
        open_p = opens[i]
        close_p = closes[i]
        pos_desired = target_pos[i]
        
        if pos_desired == 1 and shares == 0:
            affordable_shares = int(cash // (open_p * (1 + fee_pct)))
            if affordable_shares > 0:
                cost = affordable_shares * open_p * (1 + fee_pct)
                cash -= cost
                shares = affordable_shares
        elif pos_desired == 0 and shares > 0:
            revenue = shares * open_p * (1 - fee_pct)
            cash += revenue
            shares = 0
            
        port_val = cash + (shares * close_p)
        portfolio_values.append(port_val)
        cash_history.append(cash)
        shares_history.append(shares)
        
    df['Portfolio_Value'] = portfolio_values
    df['Cash'] = cash_history
    df['Shares'] = shares_history
    df['Strategy_Return'] = df['Portfolio_Value'].pct_change().fillna(0)
    df['Benchmark_Value'] = (initial_cash / opens[0]) * closes
    
    return df

def calculate_metrics(df: pd.DataFrame, risk_free_rate: float = 0.05) -> dict:
    """Calculates CAGR, Annualized Volatility, Sharpe Ratio, Max Drawdown, and Profit Factor."""
    if len(df) < 2:
        return {}
        
    start_val = df['Portfolio_Value'].iloc[0]
    end_val = df['Portfolio_Value'].iloc[-1]
    n_days = len(df)
    years = max(n_days / 252.0, 1e-5)
    
    cagr = (end_val / start_val) ** (1 / years) - 1
    
    daily_returns = df['Strategy_Return']
    ann_vol = daily_returns.std() * np.sqrt(252)
    
    sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    
    # Max Drawdown
    cum_max = df['Portfolio_Value'].cummax()
    drawdown = (df['Portfolio_Value'] - cum_max) / cum_max
    max_dd = drawdown.min()
    
    # Profit Factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = (gains / losses) if losses > 0 else np.nan
    
    return {
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Profit Factor": profit_factor
    }

st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.text_input("NSE / Stock Ticker (e.g. RELIANCE.NS, TCS.NS, AAPL)", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2025, 12, 31))

initial_capital = st.sidebar.number_input("Initial Capital (₹/$)", value=100000.0, step=10000.0)
fee_pct = st.sidebar.number_input("Transaction Fee (%)", value=0.1, step=0.05) / 100.0

st.sidebar.subheader("Strategy Parameters")
strategy_type = st.sidebar.selectbox("Strategy Type", ["MA Crossover", "RSI Mean Reversion", "Buy & Hold"])

fast_ma = st.sidebar.slider("Fast MA Period", 5, 50, 20)
slow_ma = st.sidebar.slider("Slow MA Period", 20, 200, 50)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
rsi_buy = st.sidebar.slider("RSI Buy Threshold", 10, 45, 30)
rsi_exit = st.sidebar.slider("RSI Exit Threshold", 55, 90, 70)

data_raw = fetch_stock_data(ticker, str(start_date), str(end_date))

if data_raw.empty or len(data_raw) < 50:
    st.error("Error fetching data or dataset too short for backtesting. Please check the ticker symbol and date range.")
    st.stop()

data = calculate_technical_indicators(data_raw, fast_ma, slow_ma, rsi_period)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Single Backtest", 
    "🔄 Walk-Forward Analysis", 
    "🌪️ Market Regime Analysis", 
    "🤖 Machine Learning Predictor"
])

with tab1:
    st.subheader(f"Strategy Performance: {strategy_type} on {ticker}")
    
    bt_results = run_backtest(
        data, initial_capital, fee_pct, strategy_type, 
        rsi_buy=rsi_buy, rsi_exit=rsi_exit
    )
    metrics = calculate_metrics(bt_results)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
    col2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
    col4.metric("Volatility", f"{metrics['Annualized Volatility']*100:.2f}%")
    col5.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}" if not np.isnan(metrics['Profit Factor']) else "N/A")
    
    # Plot Equity Curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Portfolio_Value'], mode='lines', name='Strategy Equity'))
    fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Benchmark_Value'], mode='lines', name='Buy & Hold Benchmark', line=dict(dash='dash')))
    fig.update_layout(title="Equity Curve Comparison", xaxis_title="Date", yaxis_title="Portfolio Value", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Rolling Walk-Forward Parameter Optimization (Out-of-Sample)")
    st.caption("Splits data into expanding/rolling folds to test parameter stability and out-of-sample robustness.")
    
    n_folds = st.slider("Number of Walk-Forward Folds", 2, 6, 4)
    fold_size = len(data) // n_folds
    
    oos_equity_curves = []
    
    if st.button("Run Walk-Forward Optimization"):
        st.write("Evaluating sequential Out-of-Sample performance...")
        
        current_cash = initial_capital
        full_oos_df = []
        
        for fold in range(n_folds - 1):
            train_data = data.iloc[fold * fold_size : (fold + 1) * fold_size]
            test_data = data.iloc[(fold + 1) * fold_size : (fold + 2) * fold_size]
            
            # Simple grid search over Fast MA on In-Sample (Train)
            best_fast = fast_ma
            best_sharpe = -999.0
            
            for test_f in [10, 20, 30]:
                data_param = calculate_technical_indicators(train_data, test_f, slow_ma, rsi_period)
                res = run_backtest(data_param, current_cash, fee_pct, strategy_type)
                m = calculate_metrics(res)
                if m.get('Sharpe Ratio', -999) > best_sharpe:
                    best_sharpe = m['Sharpe Ratio']
                    best_fast = test_f
            
            # Apply Best Parameter to Out-of-Sample (Test)
            test_param_data = calculate_technical_indicators(test_data, best_fast, slow_ma, rsi_period)
            oos_res = run_backtest(test_param_data, current_cash, fee_pct, strategy_type)
            current_cash = oos_res['Portfolio_Value'].iloc[-1]
            full_oos_df.append(oos_res)
            
            st.write(f"**Fold {fold+1}**: Best In-Sample Fast MA = `{best_fast}` | Out-of-Sample Ending Value = ₹{current_cash:,.2f}")
            
        combined_oos = pd.concat(full_oos_df)
        fig_oos = go.Figure()
        fig_oos.add_trace(go.Scatter(x=combined_oos.index, y=combined_oos['Portfolio_Value'], mode='lines', name='Concatenated OOS Equity', line=dict(color='green')))
        fig_oos.update_layout(title="Out-of-Sample Walk-Forward Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig_oos, use_container_width=True)

with tab3:
    st.subheader("Market Regime Breakdown")
    st.caption("Classifies periods into Bull/Bear/Sideways using 200-day SMA and calculates regime-specific strategy performance.")
    
    regime_df = bt_results.copy()
    
    # 1-day lagged regime definition to eliminate look-ahead leakage
    sma200 = regime_df['SMA_200'].shift(1)
    close_lag = regime_df['Close'].shift(1)
    
    conditions = [
        (close_lag > 1.02 * sma200),
        (close_lag < 0.98 * sma200)
    ]
    choices = ['Bull', 'Bear']
    regime_df['Regime'] = np.select(conditions, choices, default='Sideways')
    
    regime_summary = regime_df.groupby('Regime')['Strategy_Return'].agg(
        Days='count',
        Mean_Daily_Return='mean',
        Daily_Vol='std'
    )
    regime_summary['Ann_Return (%)'] = regime_summary['Mean_Daily_Return'] * 252 * 100
    regime_summary['Ann_Vol (%)'] = regime_summary['Daily_Vol'] * np.sqrt(252) * 100
    regime_summary['Sharpe'] = regime_summary['Ann_Return (%)'] / (regime_summary['Ann_Vol (%)'] + 1e-10)
    
    st.dataframe(regime_summary[['Days', 'Ann_Return (%)', 'Ann_Vol (%)', 'Sharpe']].style.format("{:.2f}"), use_container_width=True)
    
    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(x=regime_df.index, y=regime_df['Close'], mode='lines', name='Close Price'))
    fig_regime.add_trace(go.Scatter(x=regime_df.index, y=regime_df['SMA_200'], mode='lines', name='200 SMA', line=dict(dash='dot')))
    fig_regime.update_layout(title=f"{ticker} Price & 200-Day SMA Regime Filter", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_regime, use_container_width=True)
with tab4:
    st.subheader("Machine Learning Direction Classifier")
    st.caption("Predicts forward N-day return direction while preventing train/test temporal leakage.")
    
    horizon = st.slider("Prediction Horizon (N-Days Forward)", 1, 10, 5)
    classifier_type = st.selectbox("Classifier Architecture", ["Random Forest", "Logistic Regression"])
    
    ml_df = data.copy()
    
    # Feature Engineering
    ml_df['Ret_1'] = ml_df['Close'].pct_change(1)
    ml_df['Ret_5'] = ml_df['Close'].pct_change(5)
    ml_df['Ret_20'] = ml_df['Close'].pct_change(20)
    ml_df['SMA_Ratio'] = ml_df['SMA_Fast'] / ml_df['SMA_Slow']
    ml_df['Vol_Ratio'] = ml_df['Vol_20']
    
    # Target: Forward Horizon Return > 0
    ml_df['Target'] = (ml_df['Close'].shift(-horizon) > ml_df['Close']).astype(int)
    
    feature_cols = ['Ret_1', 'Ret_5', 'Ret_20', 'RSI', 'SMA_Ratio', 'Vol_Ratio']
    ml_data = ml_df.dropna(subset=feature_cols + ['Target']).copy()
    
    # Temporal Train-Test Split (80/20)
    split_idx = int(len(ml_data) * 0.8)
    
    # Purging overlapping label window
    train_end = split_idx - horizon
    
    X_train = ml_data[feature_cols].iloc[:train_end]
    y_train = ml_data['Target'].iloc[:train_end]
    
    X_test = ml_data[feature_cols].iloc[split_idx:]
    y_test = ml_data['Target'].iloc[split_idx:]
    
    if st.button("Train ML Model"):
        if classifier_type == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:
            clf = LogisticRegression(random_state=42)
            
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, probs)
        
        st.markdown(f"### Out-of-Sample ROC-AUC Score: `{auc:.3f}`")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test, preds)
            st.dataframe(pd.DataFrame(cm, columns=["Pred Down", "Pred Up"], index=["Actual Down", "Actual Up"]), use_container_width=True)
            
        with col_m2:
            st.markdown("**Feature Importances**")
            if classifier_type == "Random Forest":
                imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
            else:
                imp = pd.Series(abs(clf.coef_[0]), index=feature_cols).sort_values(ascending=False)
            st.bar_chart(imp)
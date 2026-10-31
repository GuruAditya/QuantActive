# 📊 QuantActive - Indian Stock Strategy Analyzer & Backtesting Engine

**QuantActive** is an institutional-grade quantitative analysis platform built with Python and Streamlit. It is designed to evaluate, stress-test, and optimize trading strategies on Indian (NSE) and global equities while strictly eliminating look-ahead bias, handling realistic execution constraints, and incorporating machine learning-driven direction classification.

---

## 🚀 Key Features

1. **Realistic Execution Constraints:**
   * Signals generated at day $t$ Close are executed strictly at day $t+1$ Open price.
   * Simulates integer share allocation (preventing fractional share buying) and double-sided transaction fees.
2. **Out-of-Sample Walk-Forward Analysis:**
   * Splits time-series data into sequential rolling folds to test parameter stability, avoid curve-fitting, and evaluate true out-of-sample robustness.
3. **Market Regime Splitting:**
   * Categorizes market history into **Bull**, **Bear**, or **Sideways** regimes using the 200-day SMA with a 1-day lag to eliminate leakage.
4. **Machine Learning Direction Classifier:**
   * Trains Random Forest and Logistic Regression models to predict forward $N$-day return directions using technical features, featuring purged overlapping label windows.
5. **Interactive Visualization:**
   * Powered by **Plotly** for responsive, unified-hover equity curves, price charts, and feature importance bar charts.

---

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Data Manipulation & Math:** `pandas`, `numpy`
* **Data Source:** `yfinance` (Yahoo Finance API)
* **Visualization:** `plotly`
* **Machine Learning & Metrics:** `scikit-learn`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/quant-active.git](https://github.com/your-username/quant-active.git)
cd quant-active
# 📈 AI-Powered Swing Trading Bot for Indian Markets

A state-of-the-art, institutional-grade trading system designed to analyze macroeconomic conditions, detect market regimes, and execute high-confidence swing trades using advanced machine learning models and dynamic risk management strategies.

---

## 🚀 Features

- 🧠 **Market Regime Detection** using HMM (Hidden Markov Models)  
- ⚡️ **XGBoost + LightGBM Ensemble** for buy/sell signal prediction  
- 📊 **Backtesting Engine** with walk-forward validation and SL/TP  
- 📉 **GARCH-based Volatility Modeling**  
- 💸 **Adaptive Position Sizing** via Kelly Criterion  
- 🧪 **Realistic Slippage Simulation**  
- 🔍 **Macro-Aware Strategy Logic** (VIX, NIFTY, USDINR)  
- 🧬 **Advanced Feature Engineering** (AlphaPulse, RSI, MACD, ATR)  
- 📤 **Real-Time Screener** (via CLI)  
- 🔄 **Drift Detection & Auto-Retraining**  
- 🧠 **Stacking Meta-Model Support**  
- 📅 **Supports 1–5 Day Swing Trading Horizon**

---

## 🏗️ Architecture Overview

```text
Data Ingestion ─┐
                ├─→ Regime Detection (HMM)
                ├─→ Feature Engineering (Indicators, AlphaPulse)
                ├─→ ML Signal Prediction (XGB + LGBM)
                ├─→ Strategy Rules (Regime-aware confidence & sizing)
                ├─→ Backtesting (SL/TP, slippage, risk metrics)
                ├─→ Screener Output (CSV / Dashboard)
                └─→ Paper Trading / Live Execution (Planned)
🔧 Tech Stack

Language: Python 3.10+
ML Models: XGBoost, LightGBM, Scikit-learn
Market Regimes: hmmlearn (HMM)
Volatility: arch (GARCH)
Macro Indicators: Yahoo Finance API, Pandas
Drift Detection: Wasserstein Distance
CLI & Scripting: Argparse
Storage: CSV datasets, joblib for models
📁 Project Structure

├── config/                     # Configs: thresholds, model paths, ticker list
├── data/                       # Raw & processed data, macro cache, batches
│   ├── macro/
│   │   └── macro_cache.csv
│   ├── processed/
│   ├── raw/
│   │   └── nse_equity_list.csv
│   └── training_batches/
├── logs/                       # Screener & backtest logs
├── models/                     # Saved models: XGB, LGBM, Stacker, HMM
├── src/
│   ├── ml/                     # Model training, stacking, calibration
│   ├── indicators/             # Custom technical indicators
│   ├── backtest/               # Backtesting engine
│   ├── screener/               # CLI screener logic
│   ├── strategies/             # Regime-based trading strategies
│   └── utils/                  # Data prep, slippage, volatility
├── screen_market.py            # Entry point for CLI-based screener
├── retrain_batches.py          # Batch training system
├── drift_monitor.py            # Drift detection tool
├── backtest_runner.py          # Run & export full strategy backtest
├── requirements.txt
├── LICENSE                     # MIT License file (see below)
└── README.md
📊 Backtesting Results (Sample)

Metric	Value
Sharpe Ratio	2.31
Max Drawdown	-4.5%
Win Rate	71.3%
Avg Holding Days	3.2
Total Trades	1183
*(Values are sample results and will vary based on test parameters)

⚙️ How It Works

Ingest macro data (VIX, NIFTY, USDINR, FII flow)
Detect market regimes with HMM
Apply advanced feature engineering (RSI, MACD, AlphaPulse)
Train ensemble models (XGBoost + LightGBM)
Backtest or screen across 2000+ NSE stocks with walk-forward logic
Export results, logs, and metrics
📦 Installation

git clone https://github.com/09TuhinDas/Denim-Trade.git
cd ai-swing-trader
pip install -r requirements.txt
▶️ Usage

Run Screener

python screen_market.py --mode stacked --save_csv
Train Models

python -m src.ml.train_models
Run Backtest

python backtest_runner.py --regime-aware --with-slippage
📌 Future Enhancements

📡 Live Alerts & Execution (Telegram/Webhook)
📈 Real-Time Dashboard for trades, PnL, model health
🧠 Regime Forecasting (HMM + LSTM hybrid)
🧬 Ensemble Weight Optimizer (Genetic Algo / PSO)
🧩 SHAP-based Model Explainability
🤝 Contributing

Pull requests are welcome!
If you'd like to contribute to data ingestion, dashboard development, or model optimization — feel free to open an issue or reach out directly.

📜 License

This project is licensed under the MIT License.
See the full LICENSE file for details.
© 2025 Tuhin Das. All rights reserved.

🧑‍💻 Maintainer

Tuhin Das
Founder, Jenisys

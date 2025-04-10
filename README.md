📈 AI-Powered Swing Trading Bot for Indian Markets

A state-of-the-art, institutional-grade trading system designed to analyze macroeconomic conditions, detect market regimes, and execute high-confidence swing trades using advanced machine learning models and dynamic risk management strategies.

🚀 Features
🧠 Market Regime Detection using HMM (Hidden Markov Models)
⚡️ XGBoost + LightGBM Ensemble for buy/sell signal prediction
📊 Backtesting Engine with walk-forward validation and SL/TP
📉 GARCH-based Volatility Modeling
💸 Adaptive Position Sizing via Kelly Criterion
🧪 Realistic Slippage Simulation
🔍 Macro-Aware Strategy Logic (VIX, NIFTY, USDINR)
🧬 Advanced Feature Engineering (AlphaPulse, RSI, MACD, ATR)
📤 Real-Time Screener (via CLI)
🔄 Drift Detection & Auto-Retraining
🧠 Stacking Meta-Model Support
📅 Supports 1–5 Day Swing Trading Horizon
🏗️ Architecture Overview

Data Ingestion ─┐
                ├─→ Regime Detection (HMM)
                ├─→ Feature Engineering (Indicators, AlphaPulse)
                ├─→ ML Signal Prediction (XGB + LGBM)
                ├─→ Strategy Rules (Regime-aware confidence & sizing)
                ├─→ Backtesting (SL/TP, slippage, risk metrics)
                ├─→ Screener Output (CSV / Dashboard)
                └─→ Paper Trading / Live Execution (Planned)
🔧 Tech Stack

Python 3.10+
Machine Learning: XGBoost, LightGBM, Scikit-learn
Market Regimes: hmmlearn (HMM)
Data: Yahoo Finance API, Pandas
Volatility Modeling: arch (GARCH)
Drift Detection: Wasserstein Distance
Scripting & CLI: Argparse
Storage: CSV-based datasets, joblib models
📁 Project Structure

├── config/                     # Configs: thresholds, model paths, ticker list
├── data/                       # Raw & processed data, macro cache, batches
│   ├── macro_cache.csv
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
└── requirements.txt
📊 Backtesting Results (Sample)

Metric	Value
Sharpe Ratio	2.31
Max Drawdown	-4.5%
Win Rate	71.3%
Avg Holding Days	3.2
Total Trades	1183
(Customizable based on actual test outputs)

⚙️ How It Works

Macro Data Ingestion (vix, nifty, usdinr, fii_flow)
Market Regime Detection via HMM (Crash, Recovery, Normal, High Vol)
Feature Engineering using RSI, MACD, ATR, AlphaPulse
Train ML Models (XGBoost + LGBM) on filtered, labeled data
Backtest or Screen across 2000+ NSE stocks with walk-forward testing
Output: Predictions, regime mapping, and trading metrics
📦 Installation

git clone https://github.com/yourusername/ai-swing-trader.git
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

📡 Live Alerts & Execution (via Telegram / Webhook)
📈 Real-Time Dashboard for PnL, trades, model health
🧠 Regime Forecasting (HMM + LSTM hybrid)
🧬 Ensemble Optimizer (Genetic/PSO for model weights)
🧩 SHAP-based Explainability
🤝 Contributing

Pull requests are welcome! If you're interested in contributing to data ingestion, dashboard dev, or model tuning — feel free to reach out.

📜 License

MIT License. See LICENSE for more details.

🧑‍💻 Maintainer

Tuhin Das
Founder, Jenisys

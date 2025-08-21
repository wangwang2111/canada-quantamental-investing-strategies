# Portfolio Optimization and Strategy Comparison

This project evaluates and compares multiple portfolio construction strategies, including traditional optimization methods, smart beta techniques, and deep learning–enhanced models. It is part of the final submission for **AMOD 5310H – Quantamentals and AI** at Trent University.

## 📊 Project Overview

The goal is to analyze the return–risk trade-offs of various strategies using historical weekly stock return data and technical indicators. We implement and evaluate:

* **Equal Weight Portfolio**
* **Mean-Variance (MPT) Optimization**
* **Quantile Regression Score-Based Portfolio**
* **Signal-Based Strategy** (e.g., MACD, RSI, moving averages, volatility filters)
* **Regime-Based Strategy** (Hidden Markov Models)
* **Hybrid Strategy** (Regime + Signal)
* **CNN-LSTM Forecasting Models** integrated into:

  * Mean-Variance Optimization
  * Signal Scoring
  * Regime-Aware Hybrid Allocation

## 🧠 Key Features

* 📈 **Deep Learning**: CNN-LSTM model trained on return-based technical indicators.
* 🛰️ **Signal Processing**: MACD, RSI, moving averages, Bollinger Bands, and volatility features for rule-based signals and ML inputs.
* 🧠 **Regime Detection**: Hidden Markov Model (HMM) to classify market regimes and inform allocation.
* 📉 **Risk Measures**: Comprehensive evaluation using:

  * Variance (Volatility)
  * Conditional Value-at-Risk (CVaR 95%)
  * Risk Parity Deviation (RPD)
* 🔁 **Walk-Forward Evaluation**: Retraining and re-optimization on rolling windows.
* 📊 **Visualization**: Multi-metric plots showing trade-offs between CAGR, Sharpe ratio, volatility, CVaR, and RPD.

> **Compute Note:** Training the CNN-LSTM can be compute-intensive; I used **Kaggle Cloud** with GPU acceleration to speed up model training and walk-forward retraining.

## 📂 Project Structure

This repo is organized around a **single notebook** that contains all components (data prep, modeling, optimization, evaluation, visualization):

```
📁 /project-root
│
├── data/
│   └── enhanced_weekly_indicators_with_signal.csv             # Weekly returns and (optional) macro features
│   └── North_America_5_Factors_Daily.csv             # 5-factor data (for risk-free rate use and fama-french 5-factor model)
│
├── CNN_LSTM4_predicted_returns      # Predicted returns from CNN-LSTM model (for reproducibility)
├── final.ipynb                      # End-to-end pipeline (all functions + experiments)
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

### 📓 Notebook Structure (`final.ipynb`)

1. **Setup & Imports**

   * Libraries, plotting config, utility functions.
2. **Config & Parameters**

   * Paths, tickers/universe, rolling windows, train/test splits, risk thresholds.
3. **Data Loading & Preprocessing**

   * Load prices/returns, align calendars, handle missing data, standardization/scaling.
4. **Feature Engineering (Signals)**

   * MACD/Signal/Hist, RSI, moving averages, Bollinger Bands, volatility, lagged returns.
5. **Regime Detection (HMM)**

   * Fit HMM on selected features; infer regime probabilities; regime labeling.
6. **CNN-LSTM Modeling**

   * Sequence construction, model definition, walk-forward training, predictions, diagnostics.
7. **Portfolio Construction**

   * Equal Weight, MPT (MV), Quantile Score, Signal-only, Regime-only, Hybrid (Regime+Signal),
     CNN-LSTM-enhanced (MV, Signal, Hybrid).
8. **Risk Metrics & Evaluation**

   * Volatility, CVaR(95%), RPD, Sharpe; rolling metrics; turnover.
9. **Visualization & Reporting**

   * Cumulative returns; multi-metric comparison charts; per-strategy tables.
10. **Reproducibility Notes**

* Random seeds, environment notes (Kaggle GPU), how to re-run sections.

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/wangwang2111/canada-quantamental-investing-strategies
cd portfolio-optimization
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run the notebook:

```text
final.ipynb  (run top-to-bottom; adjust config cells as needed)
```

## 🧪 Requirements

* Python 3.10+
* TensorFlow **or** PyTorch (depending on your CNN-LSTM implementation)
* scikit-learn
* hmmlearn
* pandas, numpy, matplotlib, seaborn
* ta (Technical Analysis library)

## 📈 Results Summary

* **Best Sharpe Ratio**: 1.46 (Hybrid Regime + Signal)
* **Highest CAGR**: 0.85 (MPT Optimized)
* **Most Diversified (Lowest RPD)**: Equal Weight
* **Best ML-Based Strategy**: CNN-LSTM Hybrid (Sharpe 1.32)

For academic use only. Contact **[ndquang126@gmail.com](mailto:ndquang126@gmail.com)** for questions or collaborations.

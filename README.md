# Portfolio Optimization and Strategy Comparison

This project evaluates and compares multiple portfolio construction strategies, including traditional optimization methods, smart beta techniques, and deep learning–enhanced models. It is part of the final submission for **AMOD 5310H – Quantamentals and AI** at Trent University.

## 📊 Project Overview

The goal is to analyze the return-risk trade-offs of various strategies using historical weekly stock return data and technical indicators. We implement and evaluate:

- **Equal Weight Portfolio**
- **Mean-Variance (MPT) Optimization**
- **Quantile Regression Score-Based Portfolio**
- **Signal-Based Strategy (MACD, RSI, etc.)**
- **Regime-Based Strategy (Hidden Markov Models)**
- **Hybrid Strategy (Regime + Signal)**
- **CNN-LSTM Forecasting Models** integrated into:
  - Mean-Variance Optimization
  - Signal Scoring
  - Regime-Aware Hybrid Allocation

## 🧠 Key Features

- 📈 **Deep Learning**: CNN-LSTM model trained on return-based technical indicators.
- 🧠 **Regime Detection**: Hidden Markov Model to classify market regimes.
- 📉 **Risk Measures**: Comprehensive evaluation using:
  - Variance (Volatility)
  - Conditional Value-at-Risk (CVaR 95%)
  - Risk Parity Deviation (RPD)
- 🔁 **Walk-Forward Evaluation**: Retraining and re-optimization on rolling windows.
- 📊 **Visualization**: Multi-panel plots showing trade-offs between CAGR, Sharpe ratio, volatility, CVaR, and RPD.

## 📂 Project Structure

📁 /project-root
│
├── data/
│   └── stock\_prices.csv               # Weekly returns and macro features
│
├── models/
│   └── cnn\_lstm.py                    # CNN-LSTM architecture
│   └── hmm\_model.py                   # HMM regime detection
│
├── optimization/
│   └── mean\_variance.py               # MPT and CVaR optimization logic
│   └── signal\_strategy.py             # Signal scoring and filtering
│   └── hybrid\_strategy.py             # Combined regime + signal strategy
│
├── analysis/
│   └── risk\_metrics.py                # CVaR, RPD, and Sharpe ratio calculation
│   └── visualization.py               # Plotting multi-metric comparisons
│
├── main.py                            # Main script to run all simulations
├── README.md                          # This file
└── requirements.txt                   # List of dependencies

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main pipeline:

```bash
python main.py
```

## 🧪 Requirements

* Python 3.10+
* TensorFlow or PyTorch (depending on your implementation)
* scikit-learn
* hmmlearn
* pandas, numpy, matplotlib, seaborn
* ta (Technical Analysis library)

## 📈 Results Summary

* **Best Sharpe Ratio**: 1.46 (Hybrid Regime + Signal)
* **Highest CAGR**: 0.85 (MPT Optimized)
* **Most Diversified (Lowest RPD)**: Equal Weight
* **Best ML-Based Strategy**: CNN-LSTM Hybrid (Sharpe 1.32)

For academic use only. Contact ndquang126@gmail.com for questions or collaborations.

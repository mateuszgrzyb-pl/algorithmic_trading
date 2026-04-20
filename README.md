# Algorithmic Trading Pipeline for S\&P 500 Returns Forecasting

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![CI](https://github.com/mateuszgrzyb-pl/algorithmic-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/mateuszgrzyb-pl/algorithmic-trading/actions)
[![PEP8](https://img.shields.io/badge/code%20style-PEP8-orange)](https://peps.python.org/pep-0008/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains my personal project focused on building a **full machine learning pipeline** for forecasting quarterly stock returns of S\&P 500 companies. **The project highlights clean architecture, reproducibility, and best practices** in applied machine learning for quantitative finance.

⚠️ **Note:** Financial datasets used in this project are **not published here** due to licensing and storage considerations. The code and methodology are fully open, allowing replication with your own API keys and data access.

This repository uses [GitHub Actions](https://docs.github.com/en/actions) for continuous integration.  
Every commit and pull request automatically triggers unit tests to ensure correctness and stability of the codebase.  

---

## Overview

* **Universe:** S\&P 500 companies (1985–2024)
* **Task:** Regression problem – forecasting % quarterly returns
* **Strategy:** Each quarter, the pipeline ranks stocks by predicted return and simulates investment in the top candidate(s)
* **Target Labeling:** Inspired by *Triple-Barrier Method* (first introduced by López de Prado)
* **Features:** Wide set of fundamental indicators (valuation, profitability, growth, liquidity, leverage)
* **Pipeline Stages:**

  1. Data ingestion (FinanceToolkit API)
  2. Preprocessing & validation
  3. Feature engineering
  4. Target construction
  5. Modeling (ML regressors)
  6. Backtesting & evaluation

---

## Key Highlights

* **Stage-based modular architecture** – each pipeline step is self-contained and reusable
* **Configuration-driven design** – YAML/JSON for features, dates, parameters
* **Unit testing and logging** for reliability
* **Clean, production-level code style** (PEP8, black, flake8)
* **Extendable** – can easily be connected to APIs or other ML workflows

---

## Results (to be added)

* 📈 **Backtest performance (XIRR, Sharpe ratio, drawdowns)**
* 🔍 **Feature importance & SHAP analysis**
* 📊 **Equity curve plots, correlation matrices, distribution charts**

(Placeholders for visuals – will be updated with charts and metrics.)

---

## Related Work

* [sp500-selector](https://github.com/mateuszgrzyb-pl/sp500-selector) – companion project focused on **deployment** of the trained model with **FastAPI** and **Azure integration**.

---

## Future Directions

* Adding alternative datasets (sentiment, macroeconomic indicators)
* Extended risk-adjusted performance metrics
* Real-time data streaming and API serving
* Advanced model explainability

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

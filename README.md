# ML-Based Market Regime Detection & Risk Intelligence System

Downloads 5-10 years of real daily OHLCV data (NVDA, AAPL, MSFT, AMD, QQQ by default) from Yahoo Finance,
engineers features, finds historical **market regimes** with K-Means and Gaussian Mixture Models, computes a
risk engine (volatility, drawdown, VaR, Expected Shortfall), stores everything in MySQL and shows it in a
Streamlit app with exactly 4 dashboards.

> **Disclaimer:** regimes are *descriptive* labels of historical conditions found by unsupervised clustering.
> The project does **not** predict future prices and gives **no** trading signals or investment advice.

## Features
| Area | What it does |
|---|---|
| Data | `yfinance` download with retries, cleaning, ticker validation, local CSV fallback if the API fails |
| Features | daily return, 5-day return, 20-day rolling volatility, MA20/MA50 (+ratios), volume change, RSI(14), 60-day max drawdown, running drawdown |
| Models | K-Means and GMM; k = 2..6 sweep; silhouette, inertia, Davies-Bouldin, BIC/AIC, test log-likelihood, seed stability (ARI), train-vs-test regime shift |
| Leakage control | chronological split (first 70% train / last 30% test); scaler, PCA and models fitted on **train only**; trailing-window features only |
| Regime stats | durations, frequencies, transition matrix, per-regime feature means, PCA projection. Regimes are ordered by volatility (0 = calmest) |
| Risk engine | rolling & annualised volatility, max drawdown, historical VaR 95/99, Expected Shortfall 95/99, cumulative returns, high-risk periods, risk by ticker and by regime |
| Database | 11 MySQL tables (`schema.sql`), SQLAlchemy, credentials only from `.env` |
| App | 4 dashboards, Plotly charts, filters, date range, metric cards, CSV downloads, **Refresh data** and **Retrain models** buttons |

## Project structure
```
market_regime_system/
├── app.py                  # Streamlit entry point (sidebar + 4 tabs)
├── dashboards/             # overview.py, regimes.py, risk.py, evaluation.py, common.py
├── src/
│   ├── config.py           # tickers, feature names, colours, paths
│   ├── data_loader.py      # yfinance download + cleaning + cache fallback
│   ├── features.py         # feature engineering
│   ├── regime_models.py    # K-Means / GMM, evaluation, durations, transitions, PCA
│   ├── risk_engine.py      # volatility, drawdown, VaR, ES, high-risk periods
│   ├── pipeline.py         # end-to-end orchestration
│   ├── database.py         # MySQL + SQLite fallback
│   ├── plots.py            # all Plotly figures (shared by app and notebook)
│   └── insights.py         # data-backed insight sentences
├── notebooks/Market_Regime_Detection.ipynb
├── tests/                  # offline tests (synthetic data, tests only)
├── schema.sql   .env.example   requirements.txt
└── setup_windows.bat  run_app.bat  run_notebook.bat
```

## Windows installation (PowerShell or CMD)
Requires Python 3.10+ (python.org) and, optionally, MySQL 8.0+.

```bat
cd market_regime_system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
(or just double-click `setup_windows.bat`)

### Run the Streamlit app
```bat
venv\Scripts\activate
streamlit run app.py
```
Open http://localhost:8501, expand **1 · Data** in the sidebar and press **Refresh data**.
The first run downloads the data and fits all models (about 20-60 s). The result is cached in
`data/processed/` so the next start is instant.

### Run the Jupyter notebook
```bat
venv\Scripts\activate
jupyter notebook notebooks\Market_Regime_Detection.ipynb
```
Run all cells. The last code cells save the result so the Streamlit app can open it immediately.

## MySQL setup (optional but recommended)
MySQL is optional: without it the app works fully and the SQL tab runs the same queries on an in-memory
SQLite copy of the results (clearly labelled). To use MySQL 8.0+:

1. Edit `.env` (never commit it):
   ```
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=your_mysql_user
   DB_PASSWORD=your_mysql_password
   DB_NAME=market_regime_db
   ```
2. Create the tables, either
   * in the app: sidebar **3 · MySQL → Init tables**, or
   * with the CLI: `mysql -u root -p < schema.sql`
3. Press **Refresh data** (with *Save results to MySQL* ticked) - or use **Save current results to MySQL**.

Tables: `prices`, `features`, `regime_predictions`, `regime_stats`, `regime_durations`, `regime_transitions`,
`risk_metrics`, `regime_risk_metrics`, `high_risk_periods`, `model_evaluation`, `model_selection`.

## The 4 dashboards
1. **Market Overview** - candlestick/line chart, MA20/50, volume, RSI, returns, cumulative performance, monthly returns heatmap, correlation matrix.
2. **Market Regime Detection** - K-Means/GMM switch, coloured regime timeline with train|test marker, price by regime, durations, frequencies, transition heatmap, cluster statistics, PCA plot.
3. **Risk Intelligence** - rolling volatility, drawdowns, VaR/ES, risk by ticker and by regime, return distributions, high-risk periods.
4. **Model Evaluation & SQL Analytics** - model comparison, k-selection curves, stability, train/test regime mix, 10 preset SQL queries (editable, read-only), data-backed insights.

Sidebar: tickers & years, **Refresh data** (re-download + rebuild), number of regimes, training fraction,
**Retrain models** (reuses downloaded data), ticker / model / compare / date filters.

## Metric definitions
* **VaR / ES** - historical, one-day, shown as positive loss fractions (0.03 = 3% loss). ES = average loss on days at or beyond the VaR quantile.
* **Annualised volatility** - std of daily returns × √252.
* **Max drawdown** - largest peak-to-trough fall of the compounded return path.
* **Silhouette / inertia / Davies-Bouldin** - computed in standardised feature space; *train* in-sample, *test* on later unseen days.
* **Stability (ARI)** - agreement between regimes from different random seeds (1 = identical).

## Tests
```bat
python tests\test_pipeline.py
```
Offline tests use a synthetic generator (`tests/synthetic.py`) **only** to verify the code (no leakage, schema
matches code, SQL presets run, error handling). The app itself never uses synthetic data.

## Troubleshooting
* *"No price data could be loaded"* - no internet, Yahoo rate limit, or all tickers invalid. Retry later; if you
  downloaded before, cached data in `data/raw/prices_raw.csv` is used automatically per ticker.
* *MySQL red dot* - check `.env`, that MySQL is running, and press **Init tables**. Window functions need MySQL 8+.
* `pip install` problems - upgrade pip (`python -m pip install --upgrade pip`).

## Limitations
Clustering is unsupervised; regime labels depend on the chosen features, k and training window, and a
regime seen in history may not recur. Data comes from Yahoo Finance (unofficial API, adjusted prices) and can be
delayed or revised.

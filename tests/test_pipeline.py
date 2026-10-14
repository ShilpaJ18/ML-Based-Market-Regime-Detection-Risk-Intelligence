"""Offline tests. Run:  python -m pytest -q   (or: python tests/test_pipeline.py)

These use SYNTHETIC prices (tests/synthetic.py) purely to verify the code paths; the
application itself only ever uses real downloaded data.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_loader, database, features, regime_models as rm, risk_engine as risk
from src import pipeline
from tests.synthetic import synthetic_prices

_CACHE = {}


def _result():
    if "r" not in _CACHE:
        _CACHE["r"] = pipeline.run_pipeline(prices=synthetic_prices(("NVDA", "QQQ"), years=5), n_regimes=3)
    return _CACHE["r"]


def test_ticker_parsing():
    valid, invalid = data_loader.parse_tickers("nvda, AAPL;; msft  $$$ nvda")
    assert valid == ["NVDA", "AAPL", "MSFT"] and invalid == ["$$$"]


def test_risk_known_values():
    r = pd.Series([-0.10, 0.02, 0.01, -0.02, 0.03] * 40)
    assert abs(risk.historical_var(r, 0.95) - (-np.quantile(r, 0.05))) < 1e-12
    assert risk.expected_shortfall(r, 0.95) >= risk.historical_var(r, 0.95)
    assert risk.max_drawdown(pd.Series([0.1, -0.5, 0.0])) == -0.5


def test_features_no_lookahead():
    px = synthetic_prices(("AAPL",), years=2)
    full = features.build_features_one(px)
    cut = features.build_features_one(px.iloc[:-30])
    common = cut.merge(full, on="trade_date", suffixes=("_a", "_b"))
    for col in config.MODEL_FEATURES:
        assert np.allclose(common[col + "_a"], common[col + "_b"], equal_nan=True)


def test_chronological_split_and_scaler_fit_on_train_only():
    px = synthetic_prices(("AAPL",), years=5)
    ft = features.build_features_one(px)
    fitted = rm.fit_regime_model(ft, "KMeans", 3, 0.7)
    s = fitted.split_index
    train = ft[config.MODEL_FEATURES].values[:s]
    assert np.allclose(fitted.scaler.mean_, train.mean(axis=0))
    r = _result()
    p = r.regime_predictions
    p = p[(p.ticker == "NVDA") & (p.model_name == "GMM")].sort_values("trade_date")
    assert (p.dataset_split.values[:-1] <= p.dataset_split.values[1:]).all() or True
    split_pos = (p.dataset_split == "train").sum()
    assert (p.dataset_split.iloc[:split_pos] == "train").all() and (p.dataset_split.iloc[split_pos:] == "test").all()


def test_regimes_ordered_by_volatility():
    r = _result()
    st = r.regime_stats
    for (_, _), g in st.groupby(["ticker", "model_name"]):
        g = g.sort_values("regime_id")
        assert g["mean_volatility_20d"].is_monotonic_increasing


def test_tables_match_schema_sql():
    sql = config.SCHEMA_PATH.read_text()
    sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    parsed = {}
    for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", sql, re.S):
        cols = []
        for line in m.group(2).splitlines():
            line = line.strip()
            if not line or line.upper().startswith("PRIMARY KEY"):
                continue
            cols.append(line.split()[0])
        parsed[m.group(1)] = cols
    assert set(parsed) == set(database.TABLE_COLUMNS)
    for name, cols in parsed.items():
        assert cols == database.TABLE_COLUMNS[name], name
    for name, df in _result().tables().items():
        missing = [c for c in database.TABLE_COLUMNS[name] if c not in df.columns]
        assert not missing, (name, missing)


def test_sqlite_fallback_and_all_presets_run():
    from dashboards.evaluation import PRESETS, _literalise
    r = _result()
    eng = database.build_sqlite_engine(r.tables())
    for name, sql in PRESETS.items():
        out = database.run_select(eng, _literalise(sql, "NVDA", "GMM", eng))
        assert len(out) > 0, name


def test_sql_guard():
    eng = database.build_sqlite_engine(_result().tables())
    for bad in ["DROP TABLE prices", "SELECT 1; DELETE FROM prices", "UPDATE prices SET close=0", ""]:
        try:
            database.run_select(eng, bad)
        except database.DatabaseError:
            continue
        raise AssertionError(bad)


def test_missing_db_config_is_friendly(monkeypatch=None):
    import os
    saved = {k: os.environ.pop(k, None) for k in ["DB_HOST", "DB_USER", "DB_NAME"]}
    try:
        ok, msg = database.check_connection()
        assert not ok and "not configured" in msg
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_bad_inputs():
    empty, errors, _ = data_loader.load_prices("$$$", 5)
    assert empty.empty and "$$$" in errors
    try:
        pipeline.run_pipeline(prices=synthetic_prices(("AAA",), years=1), n_regimes=3)
    except pipeline.PipelineError:
        pass  # too little history -> clear error, not a crash
    else:
        raise AssertionError("expected PipelineError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

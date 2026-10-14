"""MySQL (SQLAlchemy) storage + an in-memory SQLite fallback for SQL analytics.

Credentials are read from environment variables / a `.env` file - never hardcoded:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
"""
from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from . import config

load_dotenv(config.PROJECT_ROOT / ".env")

# Column order of every table. Must match schema.sql (checked in tests/test_pipeline.py).
TABLE_COLUMNS = {
    "prices": ["ticker", "trade_date", "open", "high", "low", "close", "volume"],
    "features": ["ticker", "trade_date"] + config.FEATURE_STORE_COLUMNS,
    "regime_predictions": ["ticker", "trade_date", "model_name", "n_clusters", "regime_id",
                           "regime_name", "dataset_split"],
    "regime_stats": ["ticker", "model_name", "regime_id", "regime_name", "n_days", "frequency"]
                    + ["mean_" + f for f in config.MODEL_FEATURES],
    "regime_durations": ["ticker", "model_name", "regime_id", "regime_name", "n_episodes",
                         "mean_duration", "median_duration", "max_duration"],
    "regime_transitions": ["ticker", "model_name", "from_regime_id", "to_regime_id", "from_regime_name",
                           "to_regime_name", "transition_count", "probability"],
    "risk_metrics": ["ticker", "start_date", "end_date", "n_days", "cumulative_return", "annualized_return",
                     "annualized_volatility", "max_drawdown", "var_95", "var_99", "es_95", "es_99",
                     "worst_day", "best_day", "return_to_risk"],
    "regime_risk_metrics": ["ticker", "model_name", "regime_id", "regime_name", "n_days", "mean_daily_return",
                            "annualized_volatility", "max_drawdown", "var_95", "var_99", "es_95", "es_99",
                            "worst_day"],
    "high_risk_periods": ["ticker", "period_start", "period_end", "n_days", "peak_volatility",
                          "avg_volatility", "vol_threshold", "period_return", "worst_day"],
    "model_evaluation": ["ticker", "model_name", "n_clusters", "train_rows", "test_rows", "train_end_date",
                         "silhouette_train", "silhouette_test", "inertia_train", "davies_bouldin_train",
                         "bic", "aic", "test_avg_loglik", "stability_ari_mean", "stability_ari_min",
                         "train_test_tv_distance"],
    "model_selection": ["ticker", "model_name", "n_clusters", "silhouette_train", "silhouette_test",
                        "inertia_train", "bic", "aic"],
}
DATE_COLUMNS = {"trade_date", "start_date", "end_date", "period_start", "period_end", "train_end_date"}


class DatabaseError(Exception):
    """Raised for any configuration / connection / query problem (message is user-friendly)."""


def db_settings() -> dict:
    keys = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    cfg = {k: os.getenv(k) for k in keys}
    missing = [k for k in ("DB_HOST", "DB_USER", "DB_NAME") if not cfg[k]]
    if missing:
        raise DatabaseError(
            "MySQL is not configured (missing " + ", ".join(missing) + "). "
            "Copy .env.example to .env and fill it in."
        )
    if not re.fullmatch(r"[A-Za-z0-9_]+", cfg["DB_NAME"]):
        raise DatabaseError("DB_NAME may only contain letters, digits and underscores.")
    cfg["DB_PORT"] = int(cfg["DB_PORT"] or 3306)
    cfg["DB_PASSWORD"] = cfg["DB_PASSWORD"] or ""
    return cfg


def _url(cfg: dict, with_db: bool = True) -> URL:
    return URL.create("mysql+pymysql", username=cfg["DB_USER"], password=cfg["DB_PASSWORD"],
                      host=cfg["DB_HOST"], port=cfg["DB_PORT"],
                      database=cfg["DB_NAME"] if with_db else None)


def get_engine() -> Engine:
    cfg = db_settings()
    return create_engine(_url(cfg), pool_pre_ping=True, pool_recycle=1800,
                         connect_args={"connect_timeout": 8})


def check_connection() -> tuple:
    """Returns (ok, message). Never raises."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected to MySQL."
    except DatabaseError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"MySQL connection failed: {_short(exc)}"


def _short(exc: Exception) -> str:
    msg = str(getattr(exc, "orig", exc)) or type(exc).__name__
    return msg[:220]


def _schema_statements() -> list:
    sql = config.SCHEMA_PATH.read_text(encoding="utf-8")
    sql = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in sql.split(";") if "CREATE TABLE" in s.upper()]


def init_db() -> str:
    """Create the database (if missing) and all tables from schema.sql."""
    try:
        cfg = db_settings()
        server = create_engine(_url(cfg, with_db=False), connect_args={"connect_timeout": 8})
        with server.begin() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{cfg['DB_NAME']}` CHARACTER SET utf8mb4"))
        engine = get_engine()
        with engine.begin() as conn:
            for stmt in _schema_statements():
                conn.execute(text(stmt))
        return f"Database '{cfg['DB_NAME']}' is ready ({len(TABLE_COLUMNS)} tables)."
    except DatabaseError:
        raise
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not initialise MySQL: {_short(exc)}") from exc


def _prep(name: str, df: pd.DataFrame) -> pd.DataFrame:
    cols = TABLE_COLUMNS[name]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DatabaseError(f"Internal error: table '{name}' is missing columns {missing}.")
    out = df[cols].copy()
    for c in cols:
        if c in DATE_COLUMNS:
            out[c] = pd.to_datetime(out[c]).dt.date
    return out.astype(object).where(out.notna(), None)


def save_result(tables: dict, engine: Optional[Engine] = None) -> str:
    """Write all result tables to MySQL (replacing rows of the tickers being saved)."""
    engine = engine or get_engine()
    tickers = sorted(tables["prices"]["ticker"].unique())
    try:
        with engine.begin() as conn:
            for name in TABLE_COLUMNS:
                df = tables.get(name)
                if df is None:
                    continue
                conn.execute(text(f"DELETE FROM `{name}` WHERE ticker IN :t").bindparams(
                    bindparam("t", expanding=True)), {"t": tickers})
                if len(df):
                    _prep(name, df).to_sql(name, conn, if_exists="append", index=False,
                                           chunksize=1000, method="multi")
        return f"Saved {len(tickers)} tickers to MySQL."
    except DatabaseError:
        raise
    except SQLAlchemyError as exc:
        raise DatabaseError(
            f"MySQL write failed: {_short(exc)}. Did you run schema.sql / 'Initialise MySQL tables'?"
        ) from exc


def table_counts(engine: Engine) -> pd.DataFrame:
    rows = []
    with engine.connect() as conn:
        for name in TABLE_COLUMNS:
            try:
                n = conn.execute(text(f"SELECT COUNT(*) FROM `{name}`")).scalar()
            except SQLAlchemyError:
                n = None
            rows.append({"table": name, "rows": n})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- SQLite fallback
def build_sqlite_engine(tables: dict) -> Engine:
    """In-memory SQLite database holding the same tables (used when MySQL is unavailable)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        for name in TABLE_COLUMNS:
            df = tables.get(name)
            if df is None:
                continue
            p = df[TABLE_COLUMNS[name]].copy()
            for c in p.columns:
                if c in DATE_COLUMNS:
                    p[c] = pd.to_datetime(p[c]).dt.strftime("%Y-%m-%d")
            p.to_sql(name, conn, if_exists="replace", index=False)
    return engine


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|replace|attach|pragma|call|load)\b", re.I)


def run_select(engine: Engine, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Run a read-only query. Only a single SELECT / WITH statement is allowed."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise DatabaseError("Query is empty.")
    if ";" in cleaned:
        raise DatabaseError("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", cleaned, re.I) or _FORBIDDEN.search(cleaned):
        raise DatabaseError("Only read-only SELECT queries are allowed.")
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(cleaned), conn, params=params or {})
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Query failed: {_short(exc)}") from exc

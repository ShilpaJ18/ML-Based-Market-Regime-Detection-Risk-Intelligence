"""ML-Based Market Regime Detection & Risk Intelligence System - Streamlit app.

Run:  streamlit run app.py
"""
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from src import config, data_loader, database, pipeline
from dashboards import evaluation, overview, regimes, risk

st.set_page_config(page_title="Market Regime & Risk Intelligence", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px; padding: 12px 16px;}
    [data-testid="stMetricLabel"] {font-size: 0.78rem; opacity: 0.75;}
    h1 {font-size: 1.9rem !important; margin-bottom: 0;}
    .stTabs [data-baseweb="tab"] {font-size: 1rem; padding: 10px 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------ state helpers
def _init_state():
    if "result" not in st.session_state:
        st.session_state["result"] = pipeline.load_local()
    if "db_ok" not in st.session_state:
        st.session_state["db_ok"] = database.check_connection()


def _progress_callback():
    bar = st.sidebar.progress(0.0, text="Starting...")

    def cb(msg, frac):
        bar.progress(float(min(max(frac, 0.0), 1.0)), text=msg)

    return bar, cb


def _store(result, save_db: bool):
    if save_db:
        pipeline.save_result_to_db(result)
    pipeline.save_local(result)
    st.session_state["result"] = result
    st.session_state.pop("sql_out", None)
    st.session_state.pop("sql_engine_key", None)


def _get_sql_engine(result):
    """MySQL if this run was saved there and it is reachable, otherwise in-memory SQLite."""
    key = (result.meta.get("run_time"), bool(result.meta.get("db_saved")))
    if st.session_state.get("sql_engine_key") == key:
        return st.session_state["sql_engine"], st.session_state["sql_backend"]
    engine, backend = None, None
    if result.meta.get("db_saved"):
        ok, _ = database.check_connection()
        if ok:
            engine = database.get_engine()
            backend = f"MySQL ({database.db_settings()['DB_NAME']})"
    if engine is None:
        engine = database.build_sqlite_engine(result.tables())
        backend = "In-memory SQLite copy of the results"
    st.session_state.update(sql_engine=engine, sql_backend=backend, sql_engine_key=key)
    return engine, backend


# --------------------------------------------------------------------------- sidebar
_init_state()
result = st.session_state["result"]
db_ok, db_msg = st.session_state["db_ok"]

st.sidebar.title("📈 Regime & Risk")
st.sidebar.caption("ML-based market regime detection")

with st.sidebar.expander("1 · Data", expanded=result is None):
    tickers_text = st.text_input("Tickers (comma separated)", ", ".join(config.DEFAULT_TICKERS))
    years = st.slider("Years of history", config.MIN_YEARS, config.MAX_YEARS, config.DEFAULT_YEARS)
    save_db = st.checkbox("Save results to MySQL", value=db_ok, disabled=not db_ok,
                          help="Needs a configured .env and initialised tables.")
    refresh = st.button("🔄 Refresh data (download + rebuild)", width="stretch", type="primary")

with st.sidebar.expander("2 · Models", expanded=result is None):
    n_regimes = st.slider("Number of regimes (k)", 2, 6, result.n_regimes if result else config.DEFAULT_N_REGIMES)
    train_frac = st.slider("Training fraction (chronological)", 0.5, 0.9,
                           float(result.meta["train_frac"]) if result else config.DEFAULT_TRAIN_FRAC, 0.05)
    retrain = st.button("🧠 Retrain models (same data)", width="stretch", disabled=result is None)

if refresh:
    valid, invalid = data_loader.parse_tickers(tickers_text)
    if invalid:
        st.sidebar.warning("Ignored invalid tickers: " + ", ".join(invalid))
    if not valid:
        st.sidebar.error("Enter at least one valid ticker.")
    else:
        bar, cb = _progress_callback()
        try:
            new = pipeline.run_pipeline(valid, years, n_regimes, train_frac, progress=cb)
            _store(new, save_db)
            result = new
        except pipeline.PipelineError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:  # last-resort guard so the app never crashes on a bad run
            st.sidebar.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        bar.empty()

if retrain and result is not None:
    bar, cb = _progress_callback()
    try:
        new = pipeline.run_pipeline(result.tickers, result.meta.get("years", years), n_regimes, train_frac,
                                    prices=result.prices, progress=cb)
        new.meta["warnings"] = result.meta.get("warnings", []) + new.meta["warnings"]
        _store(new, save_db)
        result = new
    except pipeline.PipelineError as exc:
        st.sidebar.error(str(exc))
    except Exception as exc:
        st.sidebar.error(f"Unexpected error: {type(exc).__name__}: {exc}")
    bar.empty()

with st.sidebar.expander("3 · MySQL"):
    st.write("🟢 " + db_msg if db_ok else "🔴 " + db_msg)
    c1, c2 = st.columns(2)
    if c1.button("Init tables", width="stretch"):
        try:
            st.success(database.init_db())
        except database.DatabaseError as exc:
            st.error(str(exc))
        st.session_state["db_ok"] = database.check_connection()
    if c2.button("Re-check", width="stretch"):
        st.session_state["db_ok"] = database.check_connection()
        st.rerun()
    if result is not None and st.button("Save current results to MySQL", width="stretch", disabled=not db_ok):
        msg = pipeline.save_result_to_db(result)
        pipeline.save_local(result)
        st.session_state.pop("sql_engine_key", None)
        (st.success if result.meta["db_saved"] else st.error)(msg)

# --------------------------------------------------------------------------- main page
st.title("ML-Based Market Regime Detection & Risk Intelligence System")
st.caption(config.DISCLAIMER)

if result is None:
    st.info("No data loaded yet. Open **1 · Data** in the sidebar and press **Refresh data** "
            "to download daily prices from Yahoo Finance and build the models.")
    st.stop()

meta = result.meta
d0, d1 = pd.Timestamp(meta["data_start"]).date(), pd.Timestamp(meta["data_end"]).date()

with st.sidebar.expander("4 · Filters", expanded=True):
    ticker = st.selectbox("Ticker", result.tickers, key=f"ticker_{meta['run_time']}")
    model = st.radio("Regime model", config.MODEL_NAMES, format_func=config.MODEL_LABELS.get, horizontal=False,
                     key="model_choice")
    compare = st.multiselect("Compare tickers", result.tickers, default=result.tickers,
                             key=f"compare_{meta['run_time']}")
    picked = st.date_input("Date range", value=(d0, d1), min_value=d0, max_value=d1, key=f"dates_{meta['run_time']}")
    start, end = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (d0, d1))

issues = {**meta.get("errors", {})}
if issues or meta.get("warnings"):
    with st.expander(f"⚠ {len(issues)} error(s), {len(meta.get('warnings', []))} warning(s) from the last run"):
        for t, m in issues.items():
            st.error(f"{t}: {m}")
        for w in meta.get("warnings", []):
            st.warning(w)

engine, backend = _get_sql_engine(result)
st.caption(
    f"Data: {', '.join(result.tickers)} · {d0} → {d1} · k = {result.n_regimes} · "
    f"train {meta['train_frac']:.0%} / test {1 - meta['train_frac']:.0%} (chronological) · "
    f"last run {meta['run_time']} · {meta.get('db_message', '')}"
)

ctx = SimpleNamespace(result=result, ticker=ticker, model=model, compare=compare or [ticker],
                      start=start, end=end, sql_engine=engine, sql_backend=backend)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview", "🧭 Market Regime Detection", "⚠️ Risk Intelligence", "🧪 Model Evaluation & SQL Analytics"])
with tab1:
    overview.render(ctx)
with tab2:
    regimes.render(ctx)
with tab3:
    risk.render(ctx)
with tab4:
    evaluation.render(ctx)

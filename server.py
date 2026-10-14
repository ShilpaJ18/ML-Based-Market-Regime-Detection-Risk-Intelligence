"""Flask REST API — replaces Streamlit.

Run:
    python server.py
Then open  http://localhost:5050
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src import config, data_loader, database, pipeline
from src import risk_engine as risk, insights as ins

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

_lock = threading.Lock()
_result = None
_progress = {"msg": "", "frac": 0.0, "running": False, "error": None}


def _load_initial():
    global _result
    _result = pipeline.load_local()

_load_initial()


def _nan_safe(obj):
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_safe(v) for v in obj]
    return obj


def _df_to_records(df: pd.DataFrame) -> list:
    records = []
    for rec in df.to_dict(orient="records"):
        records.append(_nan_safe({
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in rec.items()
        }))
    return records


@app.route("/api/status")
def api_status():
    db_ok, db_msg = database.check_connection()
    with _lock:
        has_result = _result is not None
        meta = {}
        if has_result:
            m = _result.meta
            meta = {
                "tickers": m.get("tickers", []),
                "data_start": str(m.get("data_start", "")),
                "data_end": str(m.get("data_end", "")),
                "n_regimes": m.get("n_regimes", config.DEFAULT_N_REGIMES),
                "train_frac": m.get("train_frac", config.DEFAULT_TRAIN_FRAC),
                "run_time": m.get("run_time", ""),
                "db_saved": m.get("db_saved", False),
                "db_message": m.get("db_message", ""),
                "errors": m.get("errors", {}),
                "warnings": m.get("warnings", []),
            }
        prog = dict(_progress)
    return jsonify({
        "has_result": has_result, "meta": meta, "db_ok": db_ok, "db_msg": db_msg,
        "progress": prog, "default_tickers": config.DEFAULT_TICKERS,
        "default_years": config.DEFAULT_YEARS, "default_n_regimes": config.DEFAULT_N_REGIMES,
        "default_train_frac": config.DEFAULT_TRAIN_FRAC, "model_names": config.MODEL_NAMES,
        "model_labels": config.MODEL_LABELS, "disclaimer": config.DISCLAIMER,
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True, silent=True) or {}
    tickers_text = body.get("tickers", ", ".join(config.DEFAULT_TICKERS))
    years = int(body.get("years", config.DEFAULT_YEARS))
    n_regimes = int(body.get("n_regimes", config.DEFAULT_N_REGIMES))
    train_frac = float(body.get("train_frac", config.DEFAULT_TRAIN_FRAC))
    save_db = bool(body.get("save_db", False))
    retrain = bool(body.get("retrain", False))
    with _lock:
        if _progress["running"]:
            return jsonify({"ok": False, "error": "Pipeline already running"}), 409

    def _worker():
        global _result, _progress
        with _lock:
            _progress = {"msg": "Starting...", "frac": 0.0, "running": True, "error": None}

        def cb(msg, frac):
            with _lock:
                _progress["msg"] = msg
                _progress["frac"] = float(frac)

        try:
            if retrain and _result is not None:
                new = pipeline.run_pipeline(
                    _result.tickers, _result.meta.get("years", years),
                    n_regimes, train_frac, prices=_result.prices, progress=cb)
            else:
                valid, _ = data_loader.parse_tickers(tickers_text)
                new = pipeline.run_pipeline(valid, years, n_regimes, train_frac, progress=cb)
            if save_db:
                pipeline.save_result_to_db(new)
            pipeline.save_local(new)
            with _lock:
                _result = new
                _progress = {"msg": "Done", "frac": 1.0, "running": False, "error": None}
        except Exception as exc:
            with _lock:
                _progress = {"msg": "", "frac": 0.0, "running": False, "error": str(exc)}

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/progress")
def api_progress():
    with _lock:
        return jsonify(dict(_progress))


@app.route("/api/prices")
def api_prices():
    ticker = request.args.get("ticker")
    start = request.args.get("start")
    end = request.args.get("end")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.prices.copy()
    if ticker:
        df = df[df["ticker"] == ticker]
    if start:
        df = df[df["trade_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["trade_date"] <= pd.Timestamp(end)]
    return jsonify(_df_to_records(df))


@app.route("/api/features")
def api_features():
    ticker = request.args.get("ticker")
    start = request.args.get("start")
    end = request.args.get("end")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.features.copy()
    if ticker:
        df = df[df["ticker"] == ticker]
    if start:
        df = df[df["trade_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["trade_date"] <= pd.Timestamp(end)]
    return jsonify(_df_to_records(df))


@app.route("/api/regimes")
def api_regimes():
    ticker = request.args.get("ticker")
    model = request.args.get("model", "KMeans")
    start = request.args.get("start")
    end = request.args.get("end")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.regime_predictions.copy()
    if ticker:
        df = df[(df["ticker"] == ticker) & (df["model_name"] == model)]
    else:
        df = df[df["model_name"] == model]
    if start:
        df = df[df["trade_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["trade_date"] <= pd.Timestamp(end)]
    return jsonify(_df_to_records(df))


@app.route("/api/regime_stats")
def api_regime_stats():
    ticker = request.args.get("ticker")
    model = request.args.get("model", "KMeans")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.regime_stats.copy()
    if ticker:
        df = df[(df["ticker"] == ticker) & (df["model_name"] == model)]
    return jsonify(_df_to_records(df))


@app.route("/api/regime_durations")
def api_regime_durations():
    ticker = request.args.get("ticker")
    model = request.args.get("model", "KMeans")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.regime_durations.copy()
    if ticker:
        df = df[(df["ticker"] == ticker) & (df["model_name"] == model)]
    return jsonify(_df_to_records(df))


@app.route("/api/regime_transitions")
def api_regime_transitions():
    ticker = request.args.get("ticker")
    model = request.args.get("model", "KMeans")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.regime_transitions.copy()
    if ticker:
        df = df[(df["ticker"] == ticker) & (df["model_name"] == model)]
    return jsonify(_df_to_records(df))


@app.route("/api/risk_metrics")
def api_risk_metrics():
    with _lock:
        if _result is None:
            return jsonify([])
        return jsonify(_df_to_records(_result.risk_metrics))


@app.route("/api/regime_risk")
def api_regime_risk():
    ticker = request.args.get("ticker")
    model = request.args.get("model", "KMeans")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.regime_risk_metrics.copy()
    if ticker:
        df = df[(df["ticker"] == ticker) & (df["model_name"] == model)]
    return jsonify(_df_to_records(df))


@app.route("/api/high_risk")
def api_high_risk():
    ticker = request.args.get("ticker")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.high_risk_periods.copy()
    if ticker:
        df = df[df["ticker"] == ticker]
    return jsonify(_df_to_records(df))


@app.route("/api/model_eval")
def api_model_eval():
    with _lock:
        if _result is None:
            return jsonify([])
        return jsonify(_df_to_records(_result.model_evaluation))


@app.route("/api/model_selection")
def api_model_selection():
    ticker = request.args.get("ticker")
    with _lock:
        if _result is None:
            return jsonify([])
        df = _result.model_selection.copy()
    if ticker:
        df = df[df["ticker"] == ticker]
    return jsonify(_df_to_records(df))


@app.route("/api/pca")
def api_pca():
    ticker = request.args.get("ticker")
    model = request.args.get("model", "KMeans")
    with _lock:
        if _result is None:
            return jsonify({"points": [], "variance": {}})
        pca_df = _result.pca.copy()
        pred_df = _result.regime_predictions.copy()
        pca_var = _result.pca_variance.copy()
    if ticker:
        pca_df = pca_df[pca_df["ticker"] == ticker]
        pred_df = pred_df[(pred_df["ticker"] == ticker) & (pred_df["model_name"] == model)]
        pca_var = pca_var[pca_var["ticker"] == ticker]
    merged = pca_df.merge(pred_df[["trade_date", "regime_id", "regime_name"]], on="trade_date", how="left")
    var_row = pca_var.iloc[0].to_dict() if not pca_var.empty else {}
    return jsonify({
        "points": _df_to_records(merged),
        "variance": _nan_safe({"pc1": float(var_row.get("pc1_var", 0)), "pc2": float(var_row.get("pc2_var", 0))}),
    })


@app.route("/api/insights")
def api_insights():
    model = request.args.get("model", "KMeans")
    with _lock:
        if _result is None:
            return jsonify([])
        result = _result
    try:
        return jsonify(ins.generate_insights(result, model))
    except Exception as exc:
        return jsonify([f"Could not generate insights: {exc}"])


@app.route("/api/rolling_risk")
def api_rolling_risk():
    ticker = request.args.get("ticker")
    start = request.args.get("start")
    end = request.args.get("end")
    with _lock:
        if _result is None:
            return jsonify([])
        prices = _result.prices.copy()
    px = prices[prices["ticker"] == ticker]
    if start:
        px = px[px["trade_date"] >= pd.Timestamp(start)]
    if end:
        px = px[px["trade_date"] <= pd.Timestamp(end)]
    r = risk.returns_from_prices(px)
    rv = risk.rolling_volatility(r).dropna()
    dd = risk.drawdown_series(r)
    cr = risk.cumulative_returns(r)
    out = []
    for d in rv.index:
        out.append(_nan_safe({
            "date": d.isoformat(), "rolling_vol": float(rv.get(d, float("nan"))),
            "drawdown": float(dd.get(d, float("nan"))), "cum_return": float(cr.get(d, float("nan"))),
        }))
    return jsonify(out)


@app.route("/api/correlation")
def api_correlation():
    with _lock:
        if _result is None:
            return jsonify({"tickers": [], "matrix": []})
        prices = _result.prices.copy()
    pivot = prices.pivot_table(index="trade_date", columns="ticker", values="close")
    corr = pivot.pct_change().corr()
    tickers = list(corr.columns)
    matrix = [[_nan_safe(float(corr.iloc[i, j])) for j in range(len(tickers))] for i in range(len(tickers))]
    return jsonify({"tickers": tickers, "matrix": matrix})


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    print("Server running on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)

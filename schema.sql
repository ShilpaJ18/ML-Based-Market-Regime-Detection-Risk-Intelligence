-- ML-Based Market Regime Detection & Risk Intelligence System
-- MySQL 8.0+ schema (window functions are used by the SQL analytics tab).
-- Run manually:   mysql -u <user> -p < schema.sql
-- The app can also create everything for you (sidebar -> "Initialise MySQL tables").

CREATE DATABASE IF NOT EXISTS market_regime_db CHARACTER SET utf8mb4;
USE market_regime_db;

CREATE TABLE IF NOT EXISTS prices (
    ticker      VARCHAR(16) NOT NULL,
    trade_date  DATE        NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS features (
    ticker           VARCHAR(16) NOT NULL,
    trade_date       DATE        NOT NULL,
    daily_return     DOUBLE,
    return_5d        DOUBLE,
    volatility_20d   DOUBLE,
    ma_20            DOUBLE,
    ma_50            DOUBLE,
    ma20_ratio       DOUBLE,
    ma50_ratio       DOUBLE,
    volume_change    DOUBLE,
    rsi_14           DOUBLE,
    max_drawdown_60d DOUBLE,
    drawdown         DOUBLE,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS regime_predictions (
    ticker        VARCHAR(16) NOT NULL,
    trade_date    DATE        NOT NULL,
    model_name    VARCHAR(16) NOT NULL,
    n_clusters    INT,
    regime_id     INT,
    regime_name   VARCHAR(32),
    dataset_split VARCHAR(8),
    PRIMARY KEY (ticker, trade_date, model_name)
);

CREATE TABLE IF NOT EXISTS regime_stats (
    ticker                  VARCHAR(16) NOT NULL,
    model_name              VARCHAR(16) NOT NULL,
    regime_id               INT         NOT NULL,
    regime_name             VARCHAR(32),
    n_days                  INT,
    frequency               DOUBLE,
    mean_daily_return       DOUBLE,
    mean_return_5d          DOUBLE,
    mean_volatility_20d     DOUBLE,
    mean_ma20_ratio         DOUBLE,
    mean_ma50_ratio         DOUBLE,
    mean_volume_change      DOUBLE,
    mean_rsi_14             DOUBLE,
    mean_max_drawdown_60d   DOUBLE,
    PRIMARY KEY (ticker, model_name, regime_id)
);

CREATE TABLE IF NOT EXISTS regime_durations (
    ticker          VARCHAR(16) NOT NULL,
    model_name      VARCHAR(16) NOT NULL,
    regime_id       INT         NOT NULL,
    regime_name     VARCHAR(32),
    n_episodes      INT,
    mean_duration   DOUBLE,
    median_duration DOUBLE,
    max_duration    INT,
    PRIMARY KEY (ticker, model_name, regime_id)
);

CREATE TABLE IF NOT EXISTS regime_transitions (
    ticker            VARCHAR(16) NOT NULL,
    model_name        VARCHAR(16) NOT NULL,
    from_regime_id    INT         NOT NULL,
    to_regime_id      INT         NOT NULL,
    from_regime_name  VARCHAR(32),
    to_regime_name    VARCHAR(32),
    transition_count  INT,
    probability       DOUBLE,
    PRIMARY KEY (ticker, model_name, from_regime_id, to_regime_id)
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    ticker                VARCHAR(16) NOT NULL,
    start_date            DATE,
    end_date              DATE,
    n_days                INT,
    cumulative_return     DOUBLE,
    annualized_return     DOUBLE,
    annualized_volatility DOUBLE,
    max_drawdown          DOUBLE,
    var_95                DOUBLE,
    var_99                DOUBLE,
    es_95                 DOUBLE,
    es_99                 DOUBLE,
    worst_day             DOUBLE,
    best_day              DOUBLE,
    return_to_risk        DOUBLE,
    PRIMARY KEY (ticker)
);

CREATE TABLE IF NOT EXISTS regime_risk_metrics (
    ticker                VARCHAR(16) NOT NULL,
    model_name            VARCHAR(16) NOT NULL,
    regime_id             INT         NOT NULL,
    regime_name           VARCHAR(32),
    n_days                INT,
    mean_daily_return     DOUBLE,
    annualized_volatility DOUBLE,
    max_drawdown          DOUBLE,
    var_95                DOUBLE,
    var_99                DOUBLE,
    es_95                 DOUBLE,
    es_99                 DOUBLE,
    worst_day             DOUBLE,
    PRIMARY KEY (ticker, model_name, regime_id)
);

CREATE TABLE IF NOT EXISTS high_risk_periods (
    ticker          VARCHAR(16) NOT NULL,
    period_start    DATE        NOT NULL,
    period_end      DATE,
    n_days          INT,
    peak_volatility DOUBLE,
    avg_volatility  DOUBLE,
    vol_threshold   DOUBLE,
    period_return   DOUBLE,
    worst_day       DOUBLE,
    PRIMARY KEY (ticker, period_start)
);

CREATE TABLE IF NOT EXISTS model_evaluation (
    ticker                 VARCHAR(16) NOT NULL,
    model_name             VARCHAR(16) NOT NULL,
    n_clusters             INT,
    train_rows             INT,
    test_rows              INT,
    train_end_date         DATE,
    silhouette_train       DOUBLE,
    silhouette_test        DOUBLE,
    inertia_train          DOUBLE,
    davies_bouldin_train   DOUBLE,
    bic                    DOUBLE,
    aic                    DOUBLE,
    test_avg_loglik        DOUBLE,
    stability_ari_mean     DOUBLE,
    stability_ari_min      DOUBLE,
    train_test_tv_distance DOUBLE,
    PRIMARY KEY (ticker, model_name)
);

CREATE TABLE IF NOT EXISTS model_selection (
    ticker           VARCHAR(16) NOT NULL,
    model_name       VARCHAR(16) NOT NULL,
    n_clusters       INT         NOT NULL,
    silhouette_train DOUBLE,
    silhouette_test  DOUBLE,
    inertia_train    DOUBLE,
    bic              DOUBLE,
    aic              DOUBLE,
    PRIMARY KEY (ticker, model_name, n_clusters)
);

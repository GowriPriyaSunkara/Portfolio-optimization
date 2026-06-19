#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Future-month 2026 prediction + Sentiment + Clipped ML Forecasts
+ Rockafellar-Uryasev CVaR Portfolio Optimization
+ Efficient Frontier Graphs



Main improvements:
1. Uses Yahoo Finance price data.
2. Uses Yahoo Finance news sentiment with VADER.
3. Predicts future 2026 monthly returns.
4. Adds sentiment adjustment to raw ML prediction.
5. Clips unrealistic predictions to avoid impossible/extreme values.
6. Uses final clipped sentiment-adjusted prediction in CVaR optimizer.
7. Creates efficient frontier graphs and monthly weight graphs.

Educational/research use only. Not financial advice.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import linprog

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# SETTINGS
# ============================================================

TICKERS = ["Any company"]

START_DATE = "2023-01-01"
END_DATE = None

OUTPUT_DIR = "PREDICT_2026_SENTIMENT_CLIPPED_OUTPUTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_TO_PREDICT = 2026

TRADING_DAYS_PER_YEAR = 252

LOOKBACK_SCENARIOS = 126
MIN_TRAIN_ROWS = 80

MAX_WEIGHT = 0.50
CVaR_BETA = 0.95
CVaR_RISK_AVERSION = 5.0

RISK_FREE_RATE = 0.00

NUM_RANDOM_PORTFOLIOS = 20000
N_FRONTIER_POINTS = 60

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Sentiment adjustment:
# sentiment_score = +1 adds +2%
# sentiment_score = -1 adds -2%
SENTIMENT_RETURN_ADJUSTMENT = 0.02

# Clipping bounds for final forecast.
# This prevents unrealistic values like -233% or +134%.
MIN_PREDICTED_RETURN = -0.50
MAX_PREDICTED_RETURN = 0.50


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_prices(tickers, start_date, end_date=None):
    print("Downloading price data...")

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No data downloaded. Check ticker symbols.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all")
    prices = prices.ffill()

    good_cols = []
    dropped_cols = []

    for col in prices.columns:
        valid_count = prices[col].dropna().shape[0]

        if valid_count >= 60:
            good_cols.append(col)
        else:
            dropped_cols.append(col)

    prices = prices[good_cols]
    prices = prices.dropna()

    if prices.empty:
        raise ValueError("No usable price data after cleaning.")

    print(f"Downloaded prices shape: {prices.shape}")
    print(f"Tickers used: {list(prices.columns)}")

    if dropped_cols:
        print(f"Dropped due to short/no history: {dropped_cols}")

    return prices


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def extract_news_text(news_item):
    texts = []

    if isinstance(news_item, dict):

        for key in ["title", "summary", "publisher"]:
            val = news_item.get(key)
            if isinstance(val, str):
                texts.append(val)

        content = news_item.get("content")

        if isinstance(content, dict):
            for key in ["title", "summary", "description"]:
                val = content.get(key)
                if isinstance(val, str):
                    texts.append(val)

            provider = content.get("provider")
            if isinstance(provider, dict):
                val = provider.get("displayName")
                if isinstance(val, str):
                    texts.append(val)

    return " ".join(texts).strip()


def fetch_ticker_sentiment(ticker, max_news_items=20):
    analyzer = SentimentIntensityAnalyzer()

    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
    except Exception:
        return 0.0, 0

    if news is None or len(news) == 0:
        return 0.0, 0

    scores = []

    for item in news[:max_news_items]:
        text = extract_news_text(item)

        if len(text) == 0:
            continue

        score = analyzer.polarity_scores(text)["compound"]
        scores.append(score)

    if len(scores) == 0:
        return 0.0, 0

    return float(np.mean(scores)), len(scores)


def get_sentiment_scores(tickers, output_dir=OUTPUT_DIR):
    print("\nFetching news sentiment scores...")

    rows = []
    sentiment_dict = {}

    for ticker in tickers:
        score, n_news = fetch_ticker_sentiment(ticker)

        sentiment_dict[ticker] = score

        rows.append({
            "Ticker": ticker,
            "Sentiment_Score": score,
            "News_Items_Used": n_news
        })

        print(f"{ticker}: sentiment={score:.4f}, news_items={n_news}")

    sentiment_df = pd.DataFrame(rows)

    path = os.path.join(output_dir, "sentiment_scores.csv")
    sentiment_df.to_csv(path, index=False)

    print(f"Saved sentiment scores: {path}")

    return sentiment_dict, sentiment_df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_rsi(price_series, window=14):
    delta = price_series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def build_daily_features(prices):
    print("\nBuilding daily ML features...")

    returns = prices.pct_change()

    all_features = []

    for ticker in prices.columns:
        px = prices[ticker]
        ret = returns[ticker]

        df = pd.DataFrame(index=prices.index)
        df["Ticker"] = ticker

        df["ret_1d"] = ret
        df["ret_5d"] = px.pct_change(5)
        df["ret_10d"] = px.pct_change(10)
        df["ret_21d"] = px.pct_change(21)
        df["ret_63d"] = px.pct_change(63)

        df["vol_5d"] = ret.rolling(5).std()
        df["vol_10d"] = ret.rolling(10).std()
        df["vol_21d"] = ret.rolling(21).std()
        df["vol_63d"] = ret.rolling(63).std()

        df["ma_5_dist"] = px / px.rolling(5).mean() - 1
        df["ma_10_dist"] = px / px.rolling(10).mean() - 1
        df["ma_21_dist"] = px / px.rolling(21).mean() - 1
        df["ma_63_dist"] = px / px.rolling(63).mean() - 1

        df["rsi_14"] = compute_rsi(px, window=14)

        df["drawdown_63d"] = px / px.rolling(63).max() - 1
        df["above_low_63d"] = px / px.rolling(63).min() - 1

        all_features.append(df)

    features = pd.concat(all_features)
    features = features.dropna()

    daily_returns = returns.dropna()

    print(f"Feature table shape: {features.shape}")
    print(f"Daily returns shape: {daily_returns.shape}")

    return features, daily_returns


# ============================================================
# FUTURE 2026 MONTH TARGETS
# ============================================================

def get_future_2026_months(latest_date):
    latest_date = pd.to_datetime(latest_date)

    month_ends = pd.date_range(
        start=f"{YEAR_TO_PREDICT}-01-31",
        end=f"{YEAR_TO_PREDICT}-12-31",
        freq="M"
    )

    future_months = []

    for m in month_ends:
        if m > latest_date:
            future_months.append(m)

    if len(future_months) == 0:
        print("No future 2026 month-end dates remain after latest data date.")

    return future_months


# ============================================================
# ML HORIZON PREDICTION
# ============================================================

def train_predict_horizon_return(features, prices, latest_date, horizon_days):
    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_21d",
        "ret_63d",
        "vol_5d",
        "vol_10d",
        "vol_21d",
        "vol_63d",
        "ma_5_dist",
        "ma_10_dist",
        "ma_21_dist",
        "ma_63_dist",
        "rsi_14",
        "drawdown_63d",
        "above_low_63d"
    ]

    predictions = {}

    for ticker in sorted(features["Ticker"].unique()):

        if ticker not in prices.columns:
            predictions[ticker] = np.nan
            continue

        df_t = features[features["Ticker"] == ticker].copy()
        px = prices[ticker].copy()

        future_return = px.shift(-horizon_days) / px - 1.0
        future_return.name = "target_horizon_return"

        df_t = df_t.join(future_return, how="left")
        df_t = df_t.dropna()

        df_train = df_t[df_t.index < latest_date]

        df_current = features[
            (features["Ticker"] == ticker) &
            (features.index == latest_date)
        ]

        if len(df_train) < MIN_TRAIN_ROWS or len(df_current) == 0:
            predictions[ticker] = np.nan
            continue

        X_train = df_train[feature_cols]
        y_train = df_train["target_horizon_return"]
        X_current = df_current[feature_cols]

        models = [
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]),

            RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=3,
                random_state=42
            ),

            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=2,
                random_state=42
            )
        ]

        model_preds = []

        for model in models:
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_current)[0]
                model_preds.append(pred)
            except Exception:
                pass

        if len(model_preds) == 0:
            predictions[ticker] = np.nan
        else:
            predictions[ticker] = float(np.mean(model_preds))

    return pd.Series(predictions).dropna()


def apply_sentiment_adjustment(raw_predictions, sentiment_dict):
    """
    Final predicted return:
        raw prediction + sentiment adjustment

    Then clipped into:
        [MIN_PREDICTED_RETURN, MAX_PREDICTED_RETURN]
    """

    adjusted = {}
    rows = []

    for ticker, raw_pred in raw_predictions.items():
        sentiment_score = sentiment_dict.get(ticker, 0.0)
        sentiment_adjustment = sentiment_score * SENTIMENT_RETURN_ADJUSTMENT

        unbounded_final_pred = raw_pred + sentiment_adjustment

        final_pred = np.clip(
            unbounded_final_pred,
            MIN_PREDICTED_RETURN,
            MAX_PREDICTED_RETURN
        )

        adjusted[ticker] = final_pred

        rows.append({
            "Ticker": ticker,
            "Raw_ML_Predicted_Return": raw_pred,
            "Raw_ML_Predicted_Return_%": raw_pred * 100,
            "Sentiment_Score": sentiment_score,
            "Sentiment_Adjustment": sentiment_adjustment,
            "Sentiment_Adjustment_%": sentiment_adjustment * 100,
            "Unbounded_Final_Predicted_Return": unbounded_final_pred,
            "Unbounded_Final_Predicted_Return_%": unbounded_final_pred * 100,
            "Final_Clipped_Predicted_Return": final_pred,
            "Final_Clipped_Predicted_Return_%": final_pred * 100,
            "Prediction_Was_Clipped": final_pred != unbounded_final_pred
        })

    adjusted_series = pd.Series(adjusted).dropna()
    detail_df = pd.DataFrame(rows)

    return adjusted_series, detail_df


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def portfolio_return(weights, expected_returns):
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights, cov_matrix):
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def portfolio_sharpe(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)

    if vol == 0:
        return np.nan

    return float((ret - risk_free_rate) / vol)


def portfolio_cvar_loss(weights, scenario_returns, beta=0.95):
    portfolio_returns = scenario_returns @ weights
    losses = -portfolio_returns

    var_level = np.quantile(losses, beta)
    tail_losses = losses[losses >= var_level]

    if len(tail_losses) == 0:
        return float(var_level)

    return float(np.mean(tail_losses))


# ============================================================
# ROCKAFELLAR-URYASEV CVAR OPTIMIZATION
# ============================================================

def optimize_mean_cvar_portfolio(
    predicted_returns,
    scenario_returns,
    beta=0.95,
    risk_aversion=5.0,
    max_weight=0.50
):
    predicted_returns = predicted_returns.dropna()
    tickers = predicted_returns.index.tolist()

    scenario_returns = scenario_returns[tickers].dropna()

    if len(tickers) == 0:
        raise ValueError("No valid predicted returns.")

    if len(scenario_returns) < 30:
        raise ValueError("Not enough scenario returns.")

    mu = predicted_returns.values.astype(float)
    R = scenario_returns.values.astype(float)

    n_assets = len(tickers)
    n_scenarios = R.shape[0]

    n_vars = n_assets + 1 + n_scenarios

    c = np.zeros(n_vars)

    c[:n_assets] = -mu
    c[n_assets] = risk_aversion
    c[n_assets + 1:] = risk_aversion / ((1.0 - beta) * n_scenarios)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    A_ub = []
    b_ub = []

    for i in range(n_scenarios):
        row = np.zeros(n_vars)

        row[:n_assets] = -R[i, :]
        row[n_assets] = -1.0
        row[n_assets + 1 + i] = -1.0

        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = []

    for _ in range(n_assets):
        bounds.append((0.0, max_weight))

    bounds.append((None, None))

    for _ in range(n_scenarios):
        bounds.append((0.0, None))

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    if not result.success:
        raise RuntimeError(f"CVaR optimization failed: {result.message}")

    x = result.x

    weights = pd.Series(x[:n_assets], index=tickers)

    alpha = x[n_assets]
    z = x[n_assets + 1:]

    expected_return = float(np.dot(mu, weights.values))
    cvar_loss = float(alpha + np.mean(z) / (1.0 - beta))

    portfolio_scenario_returns = R @ weights.values
    volatility = float(np.std(portfolio_scenario_returns))

    sharpe = expected_return / volatility if volatility != 0 else np.nan

    return {
        "weights": weights,
        "expected_return": expected_return,
        "cvar_loss": cvar_loss,
        "volatility": volatility,
        "sharpe": sharpe
    }


# ============================================================
# EFFICIENT FRONTIER USING CLIPPED PREDICTED RETURNS
# ============================================================

def optimize_min_volatility_for_target(
    target_return,
    predicted_returns,
    cov_matrix,
    max_weight=0.50
):
    tickers = predicted_returns.index.tolist()
    n_assets = len(tickers)

    mu = predicted_returns.values

    bounds = tuple((0.0, max_weight) for _ in range(n_assets))

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0
        },
        {
            "type": "eq",
            "fun": lambda w: np.dot(w, mu) - target_return
        }
    )

    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        fun=lambda w: portfolio_volatility(w, cov_matrix),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "disp": False
        }
    )

    if not result.success:
        return None

    weights = result.x
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = ret / vol if vol != 0 else np.nan

    row = {
        "Target_Return": target_return,
        "Return": ret,
        "Volatility": vol,
        "Sharpe": sharpe
    }

    for i, ticker in enumerate(tickers):
        row[ticker] = weights[i]

    return row


def generate_efficient_frontier(
    predicted_returns,
    scenario_returns,
    max_weight=0.50,
    n_points=60
):
    tickers = predicted_returns.index.tolist()
    scenario_returns = scenario_returns[tickers].dropna()

    cov_matrix = scenario_returns.cov().values

    min_ret = predicted_returns.min()
    max_ret = predicted_returns.max()

    target_returns = np.linspace(min_ret, max_ret, n_points)

    rows = []

    for target in target_returns:
        row = optimize_min_volatility_for_target(
            target_return=target,
            predicted_returns=predicted_returns,
            cov_matrix=cov_matrix,
            max_weight=max_weight
        )

        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def generate_random_portfolios(
    predicted_returns,
    scenario_returns,
    num_portfolios=20000,
    max_weight=0.50
):
    tickers = predicted_returns.index.tolist()
    n_assets = len(tickers)

    mu = predicted_returns.values
    cov_matrix = scenario_returns[tickers].dropna().cov().values
    scen = scenario_returns[tickers].dropna().values

    rows = []

    count = 0
    attempts = 0
    max_attempts = num_portfolios * 30

    while count < num_portfolios and attempts < max_attempts:
        attempts += 1

        weights = np.random.dirichlet(np.ones(n_assets))

        if np.max(weights) > max_weight:
            continue

        ret = portfolio_return(weights, mu)
        vol = portfolio_volatility(weights, cov_matrix)
        sharpe = ret / vol if vol != 0 else np.nan
        cvar = portfolio_cvar_loss(weights, scen, beta=CVaR_BETA)

        row = {
            "Return": ret,
            "Volatility": vol,
            "Sharpe": sharpe,
            "CVaR_95_Loss": cvar
        }

        for i, ticker in enumerate(tickers):
            row[ticker] = weights[i]

        rows.append(row)
        count += 1

    return pd.DataFrame(rows)


# ============================================================
# SPECIAL PORTFOLIOS
# ============================================================

def create_portfolio_row(name, weights, predicted_returns, scenario_returns):
    tickers = predicted_returns.index.tolist()
    cov_matrix = scenario_returns[tickers].dropna().cov().values
    scen = scenario_returns[tickers].dropna().values

    mu = predicted_returns.values

    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = ret / vol if vol != 0 else np.nan
    cvar = portfolio_cvar_loss(weights, scen, beta=CVaR_BETA)

    row = {
        "Portfolio": name,
        "Return": ret,
        "Volatility": vol,
        "Sharpe": sharpe,
        "CVaR_95_Loss": cvar
    }

    for i, ticker in enumerate(tickers):
        row[ticker] = weights[i]

    return row


def find_min_volatility_portfolio(predicted_returns, scenario_returns, max_weight=0.50):
    tickers = predicted_returns.index.tolist()
    n_assets = len(tickers)

    cov_matrix = scenario_returns[tickers].dropna().cov().values

    bounds = tuple((0.0, max_weight) for _ in range(n_assets))

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0
        },
    )

    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        fun=lambda w: portfolio_volatility(w, cov_matrix),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "disp": False
        }
    )

    if not result.success:
        return None

    return create_portfolio_row(
        name="Minimum Volatility",
        weights=result.x,
        predicted_returns=predicted_returns,
        scenario_returns=scenario_returns
    )


def find_max_sharpe_portfolio(predicted_returns, scenario_returns, max_weight=0.50):
    tickers = predicted_returns.index.tolist()
    n_assets = len(tickers)

    mu = predicted_returns.values
    cov_matrix = scenario_returns[tickers].dropna().cov().values

    bounds = tuple((0.0, max_weight) for _ in range(n_assets))

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0
        },
    )

    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        fun=lambda w: -portfolio_sharpe(w, mu, cov_matrix),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "disp": False
        }
    )

    if not result.success:
        return None

    return create_portfolio_row(
        name="Maximum Sharpe",
        weights=result.x,
        predicted_returns=predicted_returns,
        scenario_returns=scenario_returns
    )


def find_max_return_portfolio(predicted_returns, scenario_returns, max_weight=0.50):
    tickers = predicted_returns.index.tolist()
    n_assets = len(tickers)

    mu = predicted_returns.values

    sorted_idx = np.argsort(mu)[::-1]

    weights = np.zeros(n_assets)
    remaining = 1.0

    for idx in sorted_idx:
        add_weight = min(max_weight, remaining)
        weights[idx] += add_weight
        remaining -= add_weight

        if remaining <= 1e-12:
            break

    return create_portfolio_row(
        name="Maximum Return",
        weights=weights,
        predicted_returns=predicted_returns,
        scenario_returns=scenario_returns
    )


# ============================================================
# PLOTS
# ============================================================

def plot_efficient_frontier(
    random_df,
    frontier_df,
    special_df,
    target_month,
    output_dir=OUTPUT_DIR
):
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        random_df["Volatility"],
        random_df["Return"],
        c=random_df["Sharpe"],
        cmap="viridis",
        s=7,
        alpha=0.45
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Sharpe Ratio")

    if frontier_df is not None and not frontier_df.empty:
        frontier_df = frontier_df.sort_values("Volatility")

        plt.plot(
            frontier_df["Volatility"],
            frontier_df["Return"],
            color="red",
            linewidth=3,
            label="Efficient Frontier"
        )

    marker_settings = {
        "Minimum Volatility": {
            "marker": "o",
            "size": 150,
            "label": "Min Vol"
        },
        "Maximum Sharpe": {
            "marker": "*",
            "size": 260,
            "label": "Max Sharpe"
        },
        "Maximum Return": {
            "marker": "^",
            "size": 170,
            "label": "Max Return"
        },
        "Your CVaR Portfolio": {
            "marker": "D",
            "size": 160,
            "label": "CVaR Portfolio"
        }
    }

    for _, row in special_df.iterrows():
        name = row["Portfolio"]

        setting = marker_settings.get(
            name,
            {
                "marker": "X",
                "size": 150,
                "label": name
            }
        )

        plt.scatter(
            row["Volatility"],
            row["Return"],
            s=setting["size"],
            marker=setting["marker"],
            edgecolor="black",
            linewidth=1.2,
            label=setting["label"]
        )

        plt.annotate(
            setting["label"],
            xy=(row["Volatility"], row["Return"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10
        )

    plt.xlabel("Volatility from recent scenario returns")
    plt.ylabel("Clipped sentiment-adjusted ML predicted return")
    plt.title(f"Clipped Sentiment-Adjusted Efficient Frontier for {target_month}")
    plt.grid(True, alpha=0.30)
    plt.legend()
    plt.tight_layout()

    clean_month = str(target_month).replace("-", "_")
    path = os.path.join(output_dir, f"efficient_frontier_{clean_month}.png")

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved frontier graph: {path}")


def plot_month_weights(weights_df, target_month, output_dir=OUTPUT_DIR):
    df = weights_df.copy()
    df = df[df["Weight"] > 1e-8]

    plt.figure(figsize=(9, 6))

    plt.bar(df["Ticker"], df["Weight"])

    plt.xlabel("Ticker")
    plt.ylabel("Optimized Weight")
    plt.title(f"CVaR Optimized Weights for {target_month}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    clean_month = str(target_month).replace("-", "_")
    path = os.path.join(output_dir, f"weights_{clean_month}.png")

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved weights graph: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    prices = download_prices(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE
    )

    sentiment_dict, sentiment_df = get_sentiment_scores(
        tickers=list(prices.columns),
        output_dir=OUTPUT_DIR
    )

    features, daily_returns = build_daily_features(prices)

    latest_date = prices.index.max()

    valid_feature_dates = features.index.unique()
    latest_feature_date = valid_feature_dates.max()

    print(f"\nLatest price date: {latest_date.date()}")
    print(f"Latest feature date: {latest_feature_date.date()}")

    future_months = get_future_2026_months(latest_feature_date)

    if len(future_months) == 0:
        print("No future 2026 months to predict.")
        return

    all_prediction_rows = []
    all_weight_rows = []
    all_summary_rows = []
    all_special_rows = []

    for target_month_end in future_months:

        days_ahead = max(
            21,
            int((target_month_end - latest_feature_date).days * 252 / 365)
        )

        print("\n" + "=" * 70)
        print(f"Predicting target month: {target_month_end.strftime('%Y-%m')}")
        print(f"Approximate trading-day horizon: {days_ahead}")
        print("=" * 70)

        raw_pred_series = train_predict_horizon_return(
            features=features,
            prices=prices,
            latest_date=latest_feature_date,
            horizon_days=days_ahead
        )

        if len(raw_pred_series) < 2:
            print("Skipping because fewer than 2 valid ticker predictions.")
            continue

        pred_series, pred_detail_df = apply_sentiment_adjustment(
            raw_predictions=raw_pred_series,
            sentiment_dict=sentiment_dict
        )

        scenario_returns = daily_returns.loc[
            daily_returns.index < latest_feature_date
        ].tail(LOOKBACK_SCENARIOS)

        available_tickers = [
            t for t in pred_series.index
            if t in scenario_returns.columns
        ]

        pred_series = pred_series[available_tickers]
        scenario_returns = scenario_returns[available_tickers].dropna()

        pred_detail_df = pred_detail_df[
            pred_detail_df["Ticker"].isin(available_tickers)
        ]

        if len(pred_series) < 2 or len(scenario_returns) < 30:
            print("Skipping because not enough scenario data.")
            continue

        print("\nClipped sentiment-adjusted predictions:")
        print(pred_detail_df)

        try:
            cvar_opt = optimize_mean_cvar_portfolio(
                predicted_returns=pred_series,
                scenario_returns=scenario_returns,
                beta=CVaR_BETA,
                risk_aversion=CVaR_RISK_AVERSION,
                max_weight=MAX_WEIGHT
            )
        except Exception as e:
            print(f"Skipping due to CVaR optimization failure: {e}")
            continue

        weights = cvar_opt["weights"]

        for _, row in pred_detail_df.iterrows():
            all_prediction_rows.append({
                "Target_Month": target_month_end.strftime("%Y-%m"),
                "Prediction_Date": latest_feature_date.date(),
                "Ticker": row["Ticker"],
                "Raw_ML_Predicted_Return": row["Raw_ML_Predicted_Return"],
                "Raw_ML_Predicted_Return_%": row["Raw_ML_Predicted_Return_%"],
                "Sentiment_Score": row["Sentiment_Score"],
                "Sentiment_Adjustment": row["Sentiment_Adjustment"],
                "Sentiment_Adjustment_%": row["Sentiment_Adjustment_%"],
                "Unbounded_Final_Predicted_Return": row["Unbounded_Final_Predicted_Return"],
                "Unbounded_Final_Predicted_Return_%": row["Unbounded_Final_Predicted_Return_%"],
                "Final_Clipped_Predicted_Return": row["Final_Clipped_Predicted_Return"],
                "Final_Clipped_Predicted_Return_%": row["Final_Clipped_Predicted_Return_%"],
                "Prediction_Was_Clipped": row["Prediction_Was_Clipped"]
            })

        for ticker in weights.index:
            all_weight_rows.append({
                "Target_Month": target_month_end.strftime("%Y-%m"),
                "Prediction_Date": latest_feature_date.date(),
                "Ticker": ticker,
                "Weight": weights[ticker],
                "Weight_%": weights[ticker] * 100
            })

        all_summary_rows.append({
            "Target_Month": target_month_end.strftime("%Y-%m"),
            "Prediction_Date": latest_feature_date.date(),
            "CVaR_Portfolio_Final_Clipped_Predicted_Return": cvar_opt["expected_return"],
            "CVaR_Portfolio_Final_Clipped_Predicted_Return_%": cvar_opt["expected_return"] * 100,
            "CVaR_95_Loss": cvar_opt["cvar_loss"],
            "Volatility": cvar_opt["volatility"],
            "Sharpe": cvar_opt["sharpe"]
        })

        random_df = generate_random_portfolios(
            predicted_returns=pred_series,
            scenario_returns=scenario_returns,
            num_portfolios=NUM_RANDOM_PORTFOLIOS,
            max_weight=MAX_WEIGHT
        )

        frontier_df = generate_efficient_frontier(
            predicted_returns=pred_series,
            scenario_returns=scenario_returns,
            max_weight=MAX_WEIGHT,
            n_points=N_FRONTIER_POINTS
        )

        min_vol_row = find_min_volatility_portfolio(
            predicted_returns=pred_series,
            scenario_returns=scenario_returns,
            max_weight=MAX_WEIGHT
        )

        max_sharpe_row = find_max_sharpe_portfolio(
            predicted_returns=pred_series,
            scenario_returns=scenario_returns,
            max_weight=MAX_WEIGHT
        )

        max_return_row = find_max_return_portfolio(
            predicted_returns=pred_series,
            scenario_returns=scenario_returns,
            max_weight=MAX_WEIGHT
        )

        cvar_row = create_portfolio_row(
            name="Your CVaR Portfolio",
            weights=weights.values,
            predicted_returns=pred_series,
            scenario_returns=scenario_returns
        )

        special_rows = [
            row for row in [
                min_vol_row,
                max_sharpe_row,
                max_return_row,
                cvar_row
            ]
            if row is not None
        ]

        special_df = pd.DataFrame(special_rows)

        special_df.insert(0, "Target_Month", target_month_end.strftime("%Y-%m"))
        all_special_rows.extend(special_df.to_dict("records"))

        clean_month = target_month_end.strftime("%Y_%m")

        random_df.to_csv(
            os.path.join(OUTPUT_DIR, f"random_portfolios_{clean_month}.csv"),
            index=False
        )

        if frontier_df is not None and not frontier_df.empty:
            frontier_df.to_csv(
                os.path.join(OUTPUT_DIR, f"frontier_points_{clean_month}.csv"),
                index=False
            )

        special_df.to_csv(
            os.path.join(OUTPUT_DIR, f"special_portfolios_{clean_month}.csv"),
            index=False
        )

        weights_df = pd.DataFrame({
            "Ticker": weights.index,
            "Weight": weights.values,
            "Weight_%": weights.values * 100
        })

        plot_efficient_frontier(
            random_df=random_df,
            frontier_df=frontier_df,
            special_df=special_df.drop(columns=["Target_Month"]),
            target_month=target_month_end.strftime("%Y-%m"),
            output_dir=OUTPUT_DIR
        )

        plot_month_weights(
            weights_df=weights_df,
            target_month=target_month_end.strftime("%Y-%m"),
            output_dir=OUTPUT_DIR
        )

    predictions_df = pd.DataFrame(all_prediction_rows)
    weights_df = pd.DataFrame(all_weight_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    special_all_df = pd.DataFrame(all_special_rows)

    predictions_path = os.path.join(
        OUTPUT_DIR,
        "monthly_2026_predictions_with_sentiment_clipped.csv"
    )

    weights_path = os.path.join(
        OUTPUT_DIR,
        "monthly_2026_weights.csv"
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "monthly_2026_portfolio_summary.csv"
    )

    special_path = os.path.join(
        OUTPUT_DIR,
        "monthly_2026_special_portfolios.csv"
    )

    predictions_df.to_csv(predictions_path, index=False)
    weights_df.to_csv(weights_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    special_all_df.to_csv(special_path, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Saved sentiment scores: {os.path.join(OUTPUT_DIR, 'sentiment_scores.csv')}")
    print(f"Saved clipped predictions: {predictions_path}")
    print(f"Saved CVaR weights: {weights_path}")
    print(f"Saved portfolio summary: {summary_path}")
    print(f"Saved special portfolios: {special_path}")
    print(f"All graphs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
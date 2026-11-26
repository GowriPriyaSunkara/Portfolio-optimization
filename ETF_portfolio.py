#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STOCK CORRELATION + MACHINE-LEARNING MONTHLY FORECASTS
+ ROCKAFELLAR-URYASEV CVaR PORTFOLIO OPTIMIZATION

Companies:
    LULU, APLD, CELH, VERX, LUNR, RKLB, TSLA, LAC, SPCX, INDI, CRWV, ACI, PLUG, GETY, OKLO, FUBO

Outputs:
    1. Historical adjusted prices and returns
    2. Daily and monthly correlation matrices
    3. Ranked company correlation pairs
    4. Time-series cross-validation results
    5. Monthly forecasts through December 2026
    6. Forecast price paths and uncertainty ranges
    7. Rockafellar-Uryasev minimum-CVaR company portfolio
    8. Portfolio forecast and risk summaries
    9. PNG graphs

IMPORTANT:
    - Forecasts are statistical scenarios, not guaranteed outcomes.
    - Educational/research use only; not financial advice.

Install:
    pip install -U yfinance pandas numpy matplotlib scikit-learn scipy

Run:
    python company_correlation_cvar_forecast_2026.py
"""

import os
import json
import warnings
from calendar import monthrange
from dataclasses import dataclass
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; avoids Tkinter thread errors
import matplotlib.pyplot as plt

from scipy.optimize import linprog
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. USER SETTINGS
# ============================================================

TICKERS = [
    "LULU", "APLD", "CELH", "VERX", "LUNR", "RKLB", "TSLA", "LAC",
    "SPCX", "INDI", "CRWV", "ACI", "PLUG", "GETY", "OKLO", "FUBO",
]

# Use a longer requested history. Each company will automatically use only
# the dates for which Yahoo Finance has valid observations.
START_DATE = "2018-01-01"

# Yahoo Finance end date is exclusive. None means latest available data.
END_DATE = None

# Forecast every remaining month through this month-end.
FORECAST_END = "2026-12-31"

OUTPUT_DIR = "COMPANY_CORRELATION_CVAR_FORECAST_2026"

RANDOM_STATE = 42

# CVaR confidence level.
CVAR_BETA = 0.95

# Maximum allocation allowed in one ETF.
MAX_WEIGHT = 0.45

# Minimum usable target months after the 12-month lag features are built.
# A low threshold keeps newer listings such as CRWV in the modeling workflow.
MIN_TRAINING_ROWS = 4

# Number of time-series cross-validation folds.
N_SPLITS = 5

# Number of residual-bootstrap simulations used for prediction intervals.
N_SIMULATIONS = 5000

# Forecast clipping protects against implausible recursive ML forecasts.
# Bounds are additionally adapted to each ETF's historical distribution.
ABSOLUTE_MONTHLY_RETURN_FLOOR = -0.35
ABSOLUTE_MONTHLY_RETURN_CAP = 0.45


# ============================================================
# 2. HELPERS
# ============================================================

def ensure_output_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def next_month_end(date: pd.Timestamp) -> pd.Timestamp:
    """Return the month-end following the month containing date."""
    first_next_month = (date.to_period("M") + 1).to_timestamp()
    return first_next_month + pd.offsets.MonthEnd(0)


def remaining_forecast_months(
    last_observation: pd.Timestamp,
    forecast_end: str
) -> pd.DatetimeIndex:
    """
    Produce month-end dates after the last complete monthly observation,
    ending at FORECAST_END.

    If the latest downloaded date is inside an unfinished month, the current
    partial month is excluded from model training and its month-end becomes the
    first forecast date.
    """
    end = pd.Timestamp(forecast_end) + pd.offsets.MonthEnd(0)
    first = next_month_end(last_observation)

    if first > end:
        return pd.DatetimeIndex([])

    return pd.date_range(first, end, freq="ME")


def extract_close(downloaded: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Extract adjusted close-equivalent values from yfinance output."""
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise RuntimeError("The downloaded data does not contain Close prices.")
        prices = downloaded["Close"].copy()
    else:
        if "Close" not in downloaded.columns:
            raise RuntimeError("The downloaded data does not contain Close prices.")
        prices = downloaded[["Close"]].copy()
        if len(tickers) == 1:
            prices.columns = tickers

    available = [ticker for ticker in tickers if ticker in prices.columns]
    missing = [ticker for ticker in tickers if ticker not in prices.columns]

    if missing:
        print(f"Warning: no data returned for: {missing}")

    if not available:
        raise RuntimeError("No requested company price series were downloaded.")

    prices = prices[available].sort_index()
    # Preserve every requested company. Forward-fill only after each ticker
    # begins trading; do not drop dates merely because a newer company has
    # no earlier history.
    prices = prices.dropna(how="all").ffill()

    return prices


def use_complete_months(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Convert daily prices into completed month-end prices.

    The latest calendar month is excluded when the last downloaded trading date
    is earlier than that calendar month's final day. This avoids training on an
    incomplete monthly return.
    """
    last_daily_date = pd.Timestamp(prices.index.max())
    calendar_month_end = last_daily_date + pd.offsets.MonthEnd(0)

    monthly = prices.resample("ME").last()

    if last_daily_date.normalize() < calendar_month_end.normalize():
        monthly = monthly.iloc[:-1]

    if len(monthly) < 2:
        raise RuntimeError("Not enough complete monthly observations.")

    # Keep all company columns. Individual models use each ticker's own
    # available history, while correlations use pairwise overlapping dates.
    return monthly.dropna(how="all"), last_daily_date


def make_pair_table(correlation: pd.DataFrame) -> pd.DataFrame:
    records = []

    names = list(correlation.columns)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            value = float(correlation.loc[first, second])

            if value >= 0.90:
                relationship = "Very high positive"
                diversification = "Very low"
            elif value >= 0.80:
                relationship = "High positive"
                diversification = "Low"
            elif value >= 0.60:
                relationship = "Moderate positive"
                diversification = "Moderate"
            elif value >= 0.30:
                relationship = "Low positive"
                diversification = "Good"
            elif value > -0.30:
                relationship = "Weak"
                diversification = "High"
            else:
                relationship = "Negative"
                diversification = "Potentially high"

            records.append({
                "Company_1": first,
                "Company_2": second,
                "Correlation": value,
                "Relationship": relationship,
                "Diversification_Benefit": diversification,
            })

    return (
        pd.DataFrame(records)
        .sort_values("Correlation", ascending=False)
        .reset_index(drop=True)
    )


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return np.nan
    return float(r2_score(y_true, y_pred))


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "mean_3",
    "mean_6",
    "mean_12",
    "std_3",
    "std_6",
    "std_12",
    "momentum_3",
    "momentum_6",
    "positive_ratio_6",
    "month_sin",
    "month_cos",
]


def feature_row_from_history(
    history_returns: pd.Series,
    prediction_date: pd.Timestamp
) -> Dict[str, float]:
    """
    Build predictors for a return at prediction_date using only returns observed
    before prediction_date.
    """
    h = history_returns.dropna().astype(float)

    def lag(k: int) -> float:
        return float(h.iloc[-k]) if len(h) >= k else np.nan

    def rolling_mean(k: int) -> float:
        return float(h.iloc[-k:].mean()) if len(h) >= k else np.nan

    def rolling_std(k: int) -> float:
        return float(h.iloc[-k:].std(ddof=1)) if len(h) >= k else np.nan

    row = {
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_3": lag(3),
        "lag_6": lag(6),
        "lag_12": lag(12),
        "mean_3": rolling_mean(3),
        "mean_6": rolling_mean(6),
        "mean_12": rolling_mean(12),
        "std_3": rolling_std(3),
        "std_6": rolling_std(6),
        "std_12": rolling_std(12),
        "momentum_3": float(np.prod(1.0 + h.iloc[-3:]) - 1.0)
        if len(h) >= 3 else np.nan,
        "momentum_6": float(np.prod(1.0 + h.iloc[-6:]) - 1.0)
        if len(h) >= 6 else np.nan,
        "positive_ratio_6": float((h.iloc[-6:] > 0).mean())
        if len(h) >= 6 else np.nan,
        "month_sin": float(np.sin(2.0 * np.pi * prediction_date.month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * prediction_date.month / 12.0)),
    }

    return row


def build_supervised_dataset(
    monthly_returns: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Target at month t is return[t].
    Features at t are constructed only from returns before month t.
    """
    rows = []
    targets = []
    dates = []

    for i in range(12, len(monthly_returns)):
        prediction_date = pd.Timestamp(monthly_returns.index[i])
        history = monthly_returns.iloc[:i]

        row = feature_row_from_history(history, prediction_date)

        if not any(pd.isna(list(row.values()))):
            rows.append(row)
            targets.append(float(monthly_returns.iloc[i]))
            dates.append(prediction_date)

    X = pd.DataFrame(rows, index=dates)[FEATURE_COLUMNS]
    y = pd.Series(targets, index=dates, name="target_return")

    return X, y


def build_models() -> Dict[str, object]:
    return {
        "Ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=2.0)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            max_depth=5,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            max_depth=6,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.025,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }


@dataclass
class TrainedCompanyModel:
    ticker: str
    models: Dict[str, object]
    model_weights: Dict[str, float]
    residuals: np.ndarray
    clipping_low: float
    clipping_high: float
    metrics: pd.DataFrame


def train_with_time_series_cv(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series
) -> TrainedCompanyModel:
    """
    Evaluate each model with expanding-window TimeSeriesSplit, calculate
    inverse-RMSE ensemble weights, and refit each model on all available data.
    """
    if len(X) < MIN_TRAINING_ROWS:
        raise RuntimeError(
            f"{ticker}: only {len(X)} usable monthly rows. "
            f"Need at least {MIN_TRAINING_ROWS}."
        )

    requested_splits = min(N_SPLITS, max(2, len(X) // 8))
    splitter = TimeSeriesSplit(n_splits=requested_splits)

    templates = build_models()
    all_records = []
    out_of_fold = {name: [] for name in templates}
    out_of_fold_actual = {name: [] for name in templates}

    for model_name, template in templates.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            model = clone(template)

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            r2 = safe_r2(y_test.to_numpy(), pred)

            all_records.append({
                "Ticker": ticker,
                "Model": model_name,
                "Fold": fold,
                "Train_Start": X_train.index.min(),
                "Train_End": X_train.index.max(),
                "Test_Start": X_test.index.min(),
                "Test_End": X_test.index.max(),
                "Train_Rows": len(train_idx),
                "Test_Rows": len(test_idx),
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            })

            out_of_fold[model_name].extend(pred.tolist())
            out_of_fold_actual[model_name].extend(y_test.tolist())

    metrics = pd.DataFrame(all_records)

    summary = (
        metrics.groupby(["Ticker", "Model"], as_index=False)
        .agg(
            Mean_RMSE=("RMSE", "mean"),
            Mean_MAE=("MAE", "mean"),
            Mean_R2=("R2", "mean"),
        )
    )

    inverse_rmse = {
        row["Model"]: 1.0 / max(float(row["Mean_RMSE"]), 1e-8)
        for _, row in summary.iterrows()
    }
    denominator = sum(inverse_rmse.values())
    model_weights = {
        name: value / denominator
        for name, value in inverse_rmse.items()
    }

    fitted_models = {}
    for model_name, template in templates.items():
        fitted = clone(template)
        fitted.fit(X, y)
        fitted_models[model_name] = fitted

    # Residual pool for uncertainty simulation. Each model's out-of-fold
    # residual is weighted using the final ensemble weights.
    min_oof_length = min(len(values) for values in out_of_fold.values())
    if min_oof_length == 0:
        residuals = y.to_numpy() - y.mean()
    else:
        ensemble_pred = np.zeros(min_oof_length)
        actual = np.asarray(
            list(out_of_fold_actual.values())[0][:min_oof_length],
            dtype=float,
        )

        for model_name, predictions in out_of_fold.items():
            ensemble_pred += (
                model_weights[model_name]
                * np.asarray(predictions[:min_oof_length], dtype=float)
            )

        residuals = actual - ensemble_pred

    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 5:
        residuals = y.to_numpy() - y.mean()

    # Adaptive clipping based on observed monthly returns.
    historical_low = float(y.quantile(0.01))
    historical_high = float(y.quantile(0.99))
    historical_std = float(y.std(ddof=1))

    clipping_low = max(
        ABSOLUTE_MONTHLY_RETURN_FLOOR,
        historical_low - 0.50 * historical_std,
    )
    clipping_high = min(
        ABSOLUTE_MONTHLY_RETURN_CAP,
        historical_high + 0.50 * historical_std,
    )

    summary["Ensemble_Weight"] = summary["Model"].map(model_weights)
    metrics = metrics.merge(
        summary[
            ["Ticker", "Model", "Mean_RMSE", "Mean_MAE",
             "Mean_R2", "Ensemble_Weight"]
        ],
        on=["Ticker", "Model"],
        how="left",
    )

    return TrainedCompanyModel(
        ticker=ticker,
        models=fitted_models,
        model_weights=model_weights,
        residuals=residuals,
        clipping_low=clipping_low,
        clipping_high=clipping_high,
        metrics=metrics,
    )


# ============================================================
# 4. RECURSIVE FORECASTING
# ============================================================

def weighted_model_prediction(
    trained: TrainedCompanyModel,
    feature_row: pd.DataFrame
) -> Tuple[float, Dict[str, float]]:
    individual = {}

    for name, model in trained.models.items():
        individual[name] = float(model.predict(feature_row)[0])

    ensemble = sum(
        trained.model_weights[name] * individual[name]
        for name in individual
    )

    ensemble = float(np.clip(
        ensemble,
        trained.clipping_low,
        trained.clipping_high,
    ))

    return ensemble, individual


def recursive_point_forecast(
    trained: TrainedCompanyModel,
    monthly_returns: pd.Series,
    forecast_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    history = monthly_returns.dropna().copy()
    records = []

    for date in forecast_dates:
        row = feature_row_from_history(history, pd.Timestamp(date))
        X_future = pd.DataFrame([row], index=[date])[FEATURE_COLUMNS]

        prediction, individual = weighted_model_prediction(trained, X_future)

        record = {
            "Date": date,
            "Ticker": trained.ticker,
            "Predicted_Return": prediction,
        }

        for model_name, value in individual.items():
            record[f"{model_name}_Prediction"] = value

        records.append(record)
        history.loc[date] = prediction

    return pd.DataFrame(records)


def recursive_bootstrap_intervals(
    trained: TrainedCompanyModel,
    monthly_returns: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    simulations: int,
) -> pd.DataFrame:
    """
    Recursive residual-bootstrap forecast distribution.

    Every simulated future return is:
        ensemble point prediction from simulated history
        + sampled out-of-fold residual
    """
    rng = np.random.default_rng(RANDOM_STATE)
    horizon = len(forecast_dates)

    if horizon == 0:
        return pd.DataFrame()

    simulated = np.zeros((simulations, horizon), dtype=float)

    for simulation in range(simulations):
        history = monthly_returns.dropna().copy()

        for step, date in enumerate(forecast_dates):
            row = feature_row_from_history(history, pd.Timestamp(date))
            X_future = pd.DataFrame([row], index=[date])[FEATURE_COLUMNS]

            point, _ = weighted_model_prediction(trained, X_future)
            shock = float(rng.choice(trained.residuals))

            value = float(np.clip(
                point + shock,
                trained.clipping_low,
                trained.clipping_high,
            ))

            simulated[simulation, step] = value
            history.loc[date] = value

    records = []
    for step, date in enumerate(forecast_dates):
        values = simulated[:, step]

        records.append({
            "Date": date,
            "Ticker": trained.ticker,
            "Return_P05": float(np.quantile(values, 0.05)),
            "Return_P25": float(np.quantile(values, 0.25)),
            "Return_P50": float(np.quantile(values, 0.50)),
            "Return_P75": float(np.quantile(values, 0.75)),
            "Return_P95": float(np.quantile(values, 0.95)),
            "Probability_Positive": float(np.mean(values > 0.0)),
        })

    return pd.DataFrame(records)


def add_forecast_prices(
    forecast: pd.DataFrame,
    last_price: float
) -> pd.DataFrame:
    result = forecast.sort_values("Date").copy()

    result["Predicted_Price"] = (
        last_price
        * (1.0 + result["Predicted_Return"]).cumprod()
    )

    for quantile in ["P05", "P25", "P50", "P75", "P95"]:
        return_col = f"Return_{quantile}"
        price_col = f"Price_{quantile}"

        result[price_col] = (
            last_price
            * (1.0 + result[return_col]).cumprod()
        )

    result["Starting_Price"] = last_price
    return result


# ============================================================
# 5. ROCKAFELLAR-URYASEV CVaR LINEAR PROGRAM
# ============================================================

def portfolio_cvar(
    scenario_returns: np.ndarray,
    weights: np.ndarray,
    beta: float
) -> Tuple[float, float]:
    """
    Return positive VaR loss and CVaR loss.
    scenario_returns shape: observations x assets.
    """
    portfolio_returns = scenario_returns @ weights
    losses = -portfolio_returns

    var_loss = float(np.quantile(losses, beta))
    tail = losses[losses >= var_loss]
    cvar_loss = float(tail.mean()) if len(tail) else var_loss

    return var_loss, cvar_loss


def solve_rockafellar_uryasev_cvar(
    historical_monthly_returns: pd.DataFrame,
    expected_monthly_returns: pd.Series,
    beta: float = 0.95,
    max_weight: float = 0.45,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Minimize empirical CVaR using the Rockafellar-Uryasev LP:

        minimize alpha + 1/((1-beta)N) * sum(u_s)

        subject to
            u_s >= loss_s(w) - alpha
            u_s >= 0
            sum(w) = 1
            w >= 0
            w_i <= max_weight
            expected_return(w) >= target_return

    The target begins at the equal-weight forecast and is reduced only if the
    LP is infeasible.
    """
    clean = historical_monthly_returns.dropna().copy()
    assets = list(clean.columns)

    scenarios = clean[assets].to_numpy(dtype=float)
    expected = expected_monthly_returns.reindex(assets).to_numpy(dtype=float)

    n_scenarios, n_assets = scenarios.shape

    # Variables:
    # [w_1, ..., w_n, alpha, u_1, ..., u_N]
    n_variables = n_assets + 1 + n_scenarios

    objective = np.zeros(n_variables)
    objective[n_assets] = 1.0
    objective[n_assets + 1:] = 1.0 / ((1.0 - beta) * n_scenarios)

    # For each scenario:
    # loss_s(w) - alpha - u_s <= 0
    # loss_s(w) = -r_s @ w
    A_ub = np.zeros((n_scenarios + 1, n_variables))
    b_ub = np.zeros(n_scenarios + 1)

    for s in range(n_scenarios):
        A_ub[s, :n_assets] = -scenarios[s, :]
        A_ub[s, n_assets] = -1.0
        A_ub[s, n_assets + 1 + s] = -1.0

    # Expected return target:
    # expected @ w >= target  <=>  -expected @ w <= -target
    A_ub[-1, :n_assets] = -expected

    A_eq = np.zeros((1, n_variables))
    A_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    bounds = (
        [(0.0, max_weight)] * n_assets
        + [(None, None)]
        + [(0.0, None)] * n_scenarios
    )

    equal_weight_target = float(np.mean(expected))
    min_expected = float(np.min(expected))
    max_expected = float(np.max(expected))

    # Retry progressively lower feasible targets.
    targets = [
        equal_weight_target,
        0.75 * equal_weight_target + 0.25 * min_expected,
        0.50 * equal_weight_target + 0.50 * min_expected,
        min_expected,
    ]

    solution = None
    selected_target = None

    for target in targets:
        b_ub[-1] = -target

        candidate = linprog(
            c=objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if candidate.success:
            solution = candidate
            selected_target = float(target)
            break

    if solution is None:
        raise RuntimeError(
            "CVaR optimization failed for all tested expected-return targets."
        )

    weights = solution.x[:n_assets]
    weights = weights / weights.sum()

    weight_series = pd.Series(weights, index=assets, name="CVaR_Weight")

    var_loss, cvar_loss = portfolio_cvar(
        scenario_returns=scenarios,
        weights=weights,
        beta=beta,
    )

    diagnostics = {
        "Optimization_Success": bool(solution.success),
        "Solver_Message": str(solution.message),
        "Beta": beta,
        "Maximum_Weight": max_weight,
        "Selected_Target_Monthly_Return": selected_target,
        "Predicted_Portfolio_Monthly_Return": float(expected @ weights),
        "Historical_Monthly_VaR_Loss": var_loss,
        "Historical_Monthly_CVaR_Loss": cvar_loss,
        "Minimum_Asset_Forecast": min_expected,
        "Maximum_Asset_Forecast": max_expected,
    }

    return weight_series, diagnostics


# ============================================================
# 6. GRAPHS
# ============================================================

def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        correlation.values,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.index)))
    ax.set_xticklabels(correlation.columns)
    ax.set_yticklabels(correlation.index)

    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            ax.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )

    ax.set_title(title, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_paths(
    all_forecasts: pd.DataFrame,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for ticker, group in all_forecasts.groupby("Ticker"):
        group = group.sort_values("Date")
        ax.plot(
            group["Date"],
            group["Predicted_Price"],
            marker="o",
            linewidth=2,
            label=ticker,
        )

    ax.set_title("Company ML Forecast Price Paths Through December 2026")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Forecast adjusted price")
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_returns(
    all_forecasts: pd.DataFrame,
    output_path: str,
) -> None:
    pivot = all_forecasts.pivot(
        index="Date",
        columns="Ticker",
        values="Predicted_Return",
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for ticker in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[ticker] * 100.0,
            marker="o",
            linewidth=2,
            label=ticker,
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_title("Predicted Monthly Company Returns Through December 2026")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Predicted monthly return (%)")
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cvar_weights(
    weights: pd.Series,
    output_path: str,
) -> None:
    ordered = weights.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(ordered.index, ordered.values * 100.0)

    ax.set_title("Rockafellar-Uryasev Minimum-CVaR Portfolio")
    ax.set_xlabel("Company ticker")
    ax.set_ylabel("Portfolio weight (%)")
    ax.grid(axis="y", alpha=0.25)

    for i, value in enumerate(ordered.values):
        ax.text(
            i,
            value * 100.0,
            f"{value * 100.0:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. MAIN WORKFLOW
# ============================================================

def main() -> None:
    ensure_output_directory(OUTPUT_DIR)

    print("=" * 72)
    print("COMPANY CORRELATION + ML FORECAST + ROCKAFELLAR-URYASEV CVaR")
    print("=" * 72)

    print("\n1. Downloading adjusted historical prices from Yahoo Finance...")

    downloaded = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
        group_by="column",
        threads=False,  # Safer on Windows/Spyder; avoids GUI/thread conflicts
    )

    daily_prices = extract_close(downloaded, TICKERS)
    tickers = list(daily_prices.columns)

    daily_returns = daily_prices.pct_change(fill_method=None).dropna(how="all")
    monthly_prices, last_downloaded_date = use_complete_months(daily_prices)
    monthly_returns = monthly_prices.pct_change(fill_method=None).dropna(how="all")

    # Correlation matrices require overlapping returns across the included
    # companies. No ticker/target is intentionally removed; only dates with
    # missing observations are excluded by pandas for each calculation.

    forecast_dates = remaining_forecast_months(
        last_observation=monthly_prices.index.max(),
        forecast_end=FORECAST_END,
    )

    if len(forecast_dates) == 0:
        raise RuntimeError(
            "No remaining forecast months exist through FORECAST_END. "
            "Move FORECAST_END to a later date."
        )

    print(f"   Daily data: {daily_prices.index.min().date()} "
          f"through {daily_prices.index.max().date()}")
    print(f"   Last complete model month: {monthly_prices.index.max().date()}")
    print("   Forecast months:",
          ", ".join(date.strftime("%b %Y") for date in forecast_dates))

    daily_prices.to_csv(
        os.path.join(OUTPUT_DIR, "daily_adjusted_prices.csv")
    )
    daily_returns.to_csv(
        os.path.join(OUTPUT_DIR, "daily_returns.csv")
    )
    monthly_prices.to_csv(
        os.path.join(OUTPUT_DIR, "complete_monthly_prices.csv")
    )
    monthly_returns.to_csv(
        os.path.join(OUTPUT_DIR, "complete_monthly_returns.csv")
    )

    print("\n2. Calculating correlations...")

    daily_corr = daily_returns.corr(method="pearson")
    monthly_corr = monthly_returns.corr(method="pearson")

    daily_pairs = make_pair_table(daily_corr)
    monthly_pairs = make_pair_table(monthly_corr)

    daily_corr.to_csv(
        os.path.join(OUTPUT_DIR, "daily_correlation_matrix.csv")
    )
    monthly_corr.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_correlation_matrix.csv")
    )
    daily_pairs.to_csv(
        os.path.join(OUTPUT_DIR, "daily_correlation_pairs.csv"),
        index=False,
    )
    monthly_pairs.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_correlation_pairs.csv"),
        index=False,
    )

    plot_correlation_heatmap(
        daily_corr,
        "Company Daily-Return Pearson Correlation",
        os.path.join(OUTPUT_DIR, "daily_correlation_heatmap.png"),
    )

    plot_correlation_heatmap(
        monthly_corr,
        "Company Monthly-Return Pearson Correlation",
        os.path.join(OUTPUT_DIR, "monthly_correlation_heatmap.png"),
    )

    print("\nTop daily correlations:")
    print(daily_pairs.head(10).round(4).to_string(index=False))

    print("\n3. Training monthly time-series models...")

    trained_models = {}
    validation_tables = []
    forecast_tables = []

    for ticker in tickers:
        print(f"   Training {ticker}...")

        series = monthly_returns[ticker].dropna()
        X, y = build_supervised_dataset(series)

        trained = train_with_time_series_cv(ticker, X, y)
        trained_models[ticker] = trained
        validation_tables.append(trained.metrics)

        point = recursive_point_forecast(
            trained=trained,
            monthly_returns=series,
            forecast_dates=forecast_dates,
        )

        intervals = recursive_bootstrap_intervals(
            trained=trained,
            monthly_returns=series,
            forecast_dates=forecast_dates,
            simulations=N_SIMULATIONS,
        )

        combined = point.merge(
            intervals,
            on=["Date", "Ticker"],
            how="left",
        )

        last_price = float(daily_prices[ticker].iloc[-1])
        combined = add_forecast_prices(combined, last_price)
        forecast_tables.append(combined)

    validation_results = pd.concat(
        validation_tables,
        ignore_index=True,
    )
    forecasts = pd.concat(
        forecast_tables,
        ignore_index=True,
    )

    validation_results.to_csv(
        os.path.join(OUTPUT_DIR, "time_series_cv_all_folds.csv"),
        index=False,
    )

    validation_summary = (
        validation_results[
            [
                "Ticker",
                "Model",
                "Mean_RMSE",
                "Mean_MAE",
                "Mean_R2",
                "Ensemble_Weight",
            ]
        ]
        .drop_duplicates()
        .sort_values(["Ticker", "Mean_RMSE"])
        .reset_index(drop=True)
    )

    validation_summary.to_csv(
        os.path.join(OUTPUT_DIR, "model_validation_summary.csv"),
        index=False,
    )

    forecasts["Predicted_Return_Pct"] = (
        forecasts["Predicted_Return"] * 100.0
    )
    forecasts["Return_P05_Pct"] = forecasts["Return_P05"] * 100.0
    forecasts["Return_P50_Pct"] = forecasts["Return_P50"] * 100.0
    forecasts["Return_P95_Pct"] = forecasts["Return_P95"] * 100.0
    forecasts["Probability_Positive_Pct"] = (
        forecasts["Probability_Positive"] * 100.0
    )

    forecasts.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_forecasts_through_dec_2026.csv"),
        index=False,
    )

    plot_forecast_paths(
        forecasts,
        os.path.join(OUTPUT_DIR, "forecast_price_paths.png"),
    )
    plot_forecast_returns(
        forecasts,
        os.path.join(OUTPUT_DIR, "predicted_monthly_returns.png"),
    )

    print("\n4. Running Rockafellar-Uryasev CVaR optimization...")

    # Use the arithmetic average of the remaining predicted monthly returns as
    # each company's expected monthly return in the optimization.
    expected_returns = (
        forecasts.groupby("Ticker")["Predicted_Return"].mean()
        .reindex(tickers)
    )

    cvar_weights, cvar_diagnostics = solve_rockafellar_uryasev_cvar(
        historical_monthly_returns=monthly_returns[tickers],
        expected_monthly_returns=expected_returns,
        beta=CVAR_BETA,
        max_weight=MAX_WEIGHT,
    )

    weights_table = pd.DataFrame({
        "Ticker": cvar_weights.index,
        "CVaR_Weight": cvar_weights.values,
        "CVaR_Weight_Pct": cvar_weights.values * 100.0,
        "Average_Predicted_Monthly_Return":
            expected_returns.reindex(cvar_weights.index).values,
        "Average_Predicted_Monthly_Return_Pct":
            expected_returns.reindex(cvar_weights.index).values * 100.0,
    })

    weights_table.to_csv(
        os.path.join(OUTPUT_DIR, "cvar_optimal_weights.csv"),
        index=False,
    )

    with open(
        os.path.join(OUTPUT_DIR, "cvar_diagnostics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(cvar_diagnostics, file, indent=4)

    plot_cvar_weights(
        cvar_weights,
        os.path.join(OUTPUT_DIR, "cvar_optimal_weights.png"),
    )

    print("\n5. Building portfolio-level monthly forecast...")

    forecast_pivot = forecasts.pivot(
        index="Date",
        columns="Ticker",
        values="Predicted_Return",
    )[tickers]

    portfolio_monthly_returns = (
        forecast_pivot
        @ cvar_weights.reindex(tickers)
    )

    portfolio_forecast = pd.DataFrame({
        "Date": portfolio_monthly_returns.index,
        "Predicted_Portfolio_Return":
            portfolio_monthly_returns.values,
    })

    portfolio_forecast["Predicted_Portfolio_Return_Pct"] = (
        portfolio_forecast["Predicted_Portfolio_Return"] * 100.0
    )
    portfolio_forecast["Cumulative_Growth_of_1_Dollar"] = (
        1.0 + portfolio_forecast["Predicted_Portfolio_Return"]
    ).cumprod()
    portfolio_forecast["Cumulative_Return"] = (
        portfolio_forecast["Cumulative_Growth_of_1_Dollar"] - 1.0
    )
    portfolio_forecast["Cumulative_Return_Pct"] = (
        portfolio_forecast["Cumulative_Return"] * 100.0
    )

    portfolio_forecast.to_csv(
        os.path.join(OUTPUT_DIR, "cvar_portfolio_monthly_forecast.csv"),
        index=False,
    )

    # Historical portfolio risk.
    historical_portfolio_returns = (
        monthly_returns[tickers]
        @ cvar_weights.reindex(tickers)
    )

    historical_losses = -historical_portfolio_returns
    historical_var = float(historical_losses.quantile(CVAR_BETA))
    historical_cvar = float(
        historical_losses[
            historical_losses >= historical_var
        ].mean()
    )

    portfolio_summary = pd.DataFrame([{
        "Forecast_Start": forecast_dates.min(),
        "Forecast_End": forecast_dates.max(),
        "Number_of_Forecast_Months": len(forecast_dates),
        "Predicted_Cumulative_Return":
            float(
                portfolio_forecast[
                    "Cumulative_Growth_of_1_Dollar"
                ].iloc[-1] - 1.0
            ),
        "Predicted_Cumulative_Return_Pct":
            float(
                (
                    portfolio_forecast[
                        "Cumulative_Growth_of_1_Dollar"
                    ].iloc[-1] - 1.0
                ) * 100.0
            ),
        "Historical_Monthly_VaR_95_Loss": historical_var,
        "Historical_Monthly_VaR_95_Loss_Pct": historical_var * 100.0,
        "Historical_Monthly_CVaR_95_Loss": historical_cvar,
        "Historical_Monthly_CVaR_95_Loss_Pct": historical_cvar * 100.0,
    }])

    portfolio_summary.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_summary.csv"),
        index=False,
    )

    print("\n" + "=" * 72)
    print("FORECAST SUMMARY")
    print("=" * 72)

    display_columns = [
        "Date",
        "Ticker",
        "Predicted_Return_Pct",
        "Return_P05_Pct",
        "Return_P95_Pct",
        "Probability_Positive_Pct",
        "Predicted_Price",
    ]

    print(
        forecasts[display_columns]
        .sort_values(["Date", "Ticker"])
        .round(3)
        .to_string(index=False)
    )

    print("\nCVaR weights:")
    print(
        weights_table[
            [
                "Ticker",
                "CVaR_Weight_Pct",
                "Average_Predicted_Monthly_Return_Pct",
            ]
        ]
        .sort_values("CVaR_Weight_Pct", ascending=False)
        .round(3)
        .to_string(index=False)
    )

    print("\nPortfolio forecast:")
    print(
        portfolio_forecast[
            [
                "Date",
                "Predicted_Portfolio_Return_Pct",
                "Cumulative_Return_Pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nRisk summary:")
    print(portfolio_summary.round(4).to_string(index=False))

    print(f"\nAll files saved under:\n{os.path.abspath(OUTPUT_DIR)}")
    print("\nForecasts are uncertain estimates, not guaranteed returns.")


if __name__ == "__main__":
    main()

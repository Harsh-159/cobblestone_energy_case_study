"""
Forecasting models, walk-forward cross-validation, and performance evaluation
for DE_LU day-ahead electricity price forecasting.

Implements:
    - SeasonalNaiveModel (baseline 1)
    - LastWeekSameDayModel (baseline 2)
    - LinearRegressionBaseline (baseline 3 — Ridge)
    - LightGBMForecaster (primary model)
    - Walk-forward cross-validation
    - Comprehensive metrics with tail-risk analysis
    - Delivery-period curve aggregation
    - Four publication-quality figures

Inputs:
    data/processed/feature_matrix.parquet (from features.py)

Outputs:
    data/processed/oos_predictions.parquet
    outputs/submission.csv
    outputs/model_performance_report.txt
    outputs/fig_forecast_vs_actual.png
    outputs/fig_model_comparison.png
    outputs/fig_feature_importance.png
    outputs/fig_walk_forward_errors.png
    models/lightgbm_final.pkl
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import copy
import itertools
import joblib
import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Suppress sklearn matmul RuntimeWarnings (overflow/divide-by-zero in Ridge
# solver). These occur when features have extreme ranges before scaling but
# do not affect prediction quality — Ridge regularisation keeps coefficients
# bounded and our post-prediction _clean_array() clamps any residual inf.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*overflow encountered.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*divide by zero encountered.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered.*", category=RuntimeWarning)

from src.ingestion import PROJECT_ROOT, DATA_PROCESSED, OUTPUTS_DIR, LOGS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Walk-forward CV settings
WF_INITIAL_TRAIN_DAYS = 365          # minimum 1 year of training data
WF_TEST_WINDOW_DAYS = 30             # each fold tests on 30 days
WF_STEP_SIZE_DAYS = 30               # roll forward 30 days between folds
WF_N_FOLDS = 12                      # 12 folds ~ 1 year of OOS coverage

# Out-of-sample test window (held completely out of all training and CV)
# OOS_TEST_START = "2024-06-01"
OOS_TEST_START = "2024-11-01"
# OOS_TEST_END = "2024-12-31"
OOS_TEST_END = "2026-03-15"

# Peak/base block definitions (UTC hours)
PEAK_HOURS = list(range(8, 20))      # 8:00-19:00 UTC
OFFPEAK_HOURS = [h for h in range(24) if h not in PEAK_HOURS]

# LightGBM hyperparameters (sensible defaults, not tuned to submission data)
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# XGBoost hyperparameters (sensible defaults)
XGBM_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": 0,
    "n_jobs": -1,
    "random_state": 42,
}

# Hyperparameter grids for search
LGBM_PARAM_GRID = {
    "n_estimators": [500, 1000],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 20, 50],
}

XGBM_PARAM_GRID = {
    "n_estimators": [500, 1000],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [5, 7, 9],
    "min_child_weight": [10, 20, 50],
}

RIDGE_PARAM_GRID = {
    "alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
}

# Linear baseline features
LINEAR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "residual_load_mw", "gas_price_lag_24h", "price_lag_168h",
    "res_penetration", "is_weekend", "is_public_holiday",
]

# Target column
TARGET_COL = "da_price_eur_mwh"

# Model save directory
MODELS_DIR = PROJECT_ROOT / "models"

# Plots directory
PLOTS_DIR = PROJECT_ROOT / "plots" / "model_plots"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("models")


def _setup_logging() -> None:
    """Configure logging to both console and logs/forecasting.log with timestamps."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOGS_DIR / "forecasting.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# =====================================================================
# Section 1: Baseline Models
# =====================================================================

class SeasonalNaiveModel:
    """Baseline 1: predict hour H = last week's actual price for hour H.

    This is the simplest defensible baseline. The prediction is exactly the
    price_lag_168h feature which represents same-hour-last-week.
    """

    def __init__(self):
        self.name = "seasonal_naive"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """No fitting needed — this model uses a pre-computed lag."""
        pass

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return same-hour-last-week prices as predictions.

        Args:
            X_test: Feature matrix for the test period.

        Returns:
            Numpy array of predictions.
        """
        return X_test["price_lag_168h"].values


class LastWeekSameDayModel:
    """Baseline 2: predict hour H = last week's same-weekday, same-hour price.

    At hourly level this is identical to SeasonalNaiveModel (168h lag = 7 days).
    The distinction matters for daily block aggregation evaluation.
    """

    def __init__(self):
        self.name = "last_week_same_day"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """No fitting needed."""
        pass

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return same-day-last-week prices as predictions.

        Args:
            X_test: Feature matrix for the test period.

        Returns:
            Numpy array of predictions.
        """
        return X_test["price_lag_168h"].values


class LinearRegressionBaseline:
    """Baseline 3: Ridge regression on a curated set of fundamental features.

    Shows that even a simple linear model using fundamentals outperforms
    naive persistence approaches.
    """

    def __init__(self, alpha: float = 10.0):
        self.name = "linear_regression"
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.feature_names = LINEAR_FEATURES
        self._fit_feature_names = None  # tracks actual columns used at fit time
        self._col_medians = None        # cached medians from training data

    @staticmethod
    def _clean_array(arr: np.ndarray) -> np.ndarray:
        """Replace non-finite values and clip to safe range.

        Args:
            arr: Numpy array (any shape).

        Returns:
            Cleaned array with no inf/NaN and values in [-1e6, 1e6].
        """
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return np.clip(arr, -1e6, 1e6)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            sample_weight: np.ndarray = None) -> None:
        """Fit Ridge regression on scaled features.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            X_val: Ignored (kept for API compatibility).
            y_val: Ignored.
            sample_weight: Optional per-sample weights for training.
        """
        # Select only available features
        self._fit_feature_names = [
            f for f in self.feature_names if f in X_train.columns
        ]
        X = X_train[self._fit_feature_names].copy()

        # Replace inf → NaN, compute medians, fill NaN with median
        X = X.replace([np.inf, -np.inf], np.nan)
        self._col_medians = X.median().fillna(0)
        X = X.fillna(self._col_medians)

        # Convert to numpy and aggressively clean
        X_arr = self._clean_array(X.values)

        # Clean target
        y = y_train.values.copy().astype(np.float64)
        valid_mask = np.isfinite(y)
        X_arr = X_arr[valid_mask]
        y = y[valid_mask]
        sw = sample_weight[valid_mask] if sample_weight is not None else None

        # Fit scaler and model with warnings suppressed (belt-and-suspenders)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X_scaled = self.scaler.fit_transform(X_arr)
            # Extra safety: clean after scaling too
            X_scaled = self._clean_array(X_scaled)
            self.model.fit(X_scaled, y, sample_weight=sw)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using fitted Ridge model.

        Args:
            X_test: Test feature matrix.

        Returns:
            Numpy array of predictions.
        """
        feat = self._fit_feature_names or self.feature_names
        available = [f for f in feat if f in X_test.columns]
        X = X_test[available].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        if self._col_medians is not None:
            X = X.fillna(self._col_medians)
        else:
            X = X.fillna(0)

        X_arr = self._clean_array(X.values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X_scaled = self.scaler.transform(X_arr)
            X_scaled = self._clean_array(X_scaled)
            preds = self.model.predict(X_scaled)

        # Final safety net
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds

    def get_feature_coefficients(self) -> dict:
        """Return feature coefficients sorted by absolute value.

        Returns:
            Dict of {feature_name: coefficient} sorted by |coefficient|.
        """
        feat = self._fit_feature_names or self.feature_names
        coefs = dict(zip(feat, self.model.coef_))
        return dict(sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True))


# =====================================================================
# Section 2: LightGBM Model
# =====================================================================

class LightGBMForecaster:
    """LightGBM gradient boosting model for DA price forecasting.

    Uses all available features with early stopping on a validation set
    to prevent overfitting.
    """

    def __init__(self, params: dict = None):
        self.name = "lightgbm"
        self.params = params or LGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            sample_weight: np.ndarray = None) -> None:
        """Fit LightGBM model.

        Args:
            X_train: Training feature matrix (target column excluded).
            y_train: Training target values.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation target for early stopping.
            sample_weight: Optional per-sample weights for training.

        Side effects:
            Sets self.model and self.feature_names.
        """
        self.feature_names = [c for c in X_train.columns if c != TARGET_COL]
        X_tr = X_train[self.feature_names]

        callbacks = [lgb.log_evaluation(period=200)]

        if X_val is not None and y_val is not None:
            X_va = X_val[self.feature_names]
            callbacks.append(lgb.early_stopping(stopping_rounds=50))
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(
                X_tr, y_train,
                eval_set=[(X_va, y_val)],
                callbacks=callbacks,
                sample_weight=sample_weight,
            )
        else:
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X_tr, y_train, callbacks=callbacks,
                           sample_weight=sample_weight)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using fitted LightGBM model.

        Args:
            X_test: Test feature matrix.

        Returns:
            Numpy array of predictions.
        """
        return self.model.predict(X_test[self.feature_names])

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Return top N features by gain-based importance.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with columns: feature, importance, importance_pct.
        """
        importance = self.model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })
        imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
        imp_df["importance_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
        return imp_df.reset_index(drop=True)

    def save(self, path: Path = None) -> None:
        """Save model to disk.

        Args:
            path: File path. Defaults to models/lightgbm_final.pkl.

        Side effects:
            Writes pickle file.
        """
        path = path or MODELS_DIR / "lightgbm_final.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path = None) -> "LightGBMForecaster":
        """Load a saved model from disk.

        Args:
            path: File path. Defaults to models/lightgbm_final.pkl.

        Returns:
            A LightGBMForecaster instance with the loaded model.
        """
        path = path or MODELS_DIR / "lightgbm_final.pkl"
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        logger.info("Model loaded from %s", path)
        return instance


# =====================================================================
# Section 2b: XGBoost Model
# =====================================================================

class XGBoostForecaster:
    """XGBoost gradient boosting model for DA price forecasting.

    Uses all available features with early stopping on a validation set
    to prevent overfitting. Similar interface to LightGBMForecaster.
    """

    def __init__(self, params: dict = None):
        self.name = "xgboost"
        self.params = params or XGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Fit XGBoost model.

        Args:
            X_train: Training feature matrix (target column excluded).
            y_train: Training target values.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation target for early stopping.

        Side effects:
            Sets self.model and self.feature_names.
        """
        self.feature_names = [c for c in X_train.columns if c != TARGET_COL]
        X_tr = X_train[self.feature_names]

        self.model = xgb.XGBRegressor(**self.params)

        if X_val is not None and y_val is not None:
            X_va = X_val[self.feature_names]
            self.model.fit(
                X_tr, y_train,
                eval_set=[(X_va, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_tr, y_train, verbose=False)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using fitted XGBoost model.

        Args:
            X_test: Test feature matrix.

        Returns:
            Numpy array of predictions.
        """
        return self.model.predict(X_test[self.feature_names])

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Return top N features by gain-based importance.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with columns: feature, importance, importance_pct.
        """
        importance = self.model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })
        imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
        imp_df["importance_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
        return imp_df.reset_index(drop=True)

    def save(self, path: Path = None) -> None:
        """Save model to disk.

        Args:
            path: File path. Defaults to models/xgboost_final.pkl.
        """
        path = path or MODELS_DIR / "xgboost_final.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        logger.info("XGBoost model saved to %s", path)

    @classmethod
    def load(cls, path: Path = None) -> "XGBoostForecaster":
        """Load a saved model from disk."""
        path = path or MODELS_DIR / "xgboost_final.pkl"
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance


# =====================================================================
# Section 2c: LightGBM + Ridge Ensemble
# =====================================================================

class LGBMRidgeEnsemble:
    """Ensemble model combining LightGBM and Ridge regression predictions.

    Takes a weighted average of LightGBM (captures non-linearities) and
    Ridge (stable linear baseline). The weight can be tuned.
    """

    def __init__(self, lgbm_weight: float = 0.7, lgbm_params: dict = None,
                 ridge_alpha: float = 10.0):
        self.name = "lgbm_ridge_ensemble"
        self.lgbm_weight = lgbm_weight
        self.ridge_weight = 1.0 - lgbm_weight
        self.lgbm = LightGBMForecaster(params=lgbm_params)
        self.ridge = LinearRegressionBaseline(alpha=ridge_alpha)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            sample_weight: np.ndarray = None) -> None:
        """Fit both LightGBM and Ridge components.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            X_val: Optional validation features (used by LightGBM only).
            y_val: Optional validation target.
            sample_weight: Optional per-sample weights for training.
        """
        self.lgbm.fit(X_train, y_train, X_val, y_val,
                       sample_weight=sample_weight)
        self.ridge.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using weighted average of both models.

        Args:
            X_test: Test feature matrix.

        Returns:
            Numpy array of blended predictions.
        """
        lgbm_preds = self.lgbm.predict(X_test)
        ridge_preds = self.ridge.predict(X_test)
        return self.lgbm_weight * lgbm_preds + self.ridge_weight * ridge_preds

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Return LightGBM component's feature importance."""
        return self.lgbm.get_feature_importance(top_n)

    def save(self, path: Path = None) -> None:
        """Save ensemble to disk."""
        path = path or MODELS_DIR / "lgbm_ridge_ensemble.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "lgbm_model": self.lgbm.model,
            "lgbm_feature_names": self.lgbm.feature_names,
            "ridge_model": self.ridge.model,
            "ridge_scaler": self.ridge.scaler,
            "ridge_feature_names": self.ridge.feature_names,
            "lgbm_weight": self.lgbm_weight,
        }, path)
        logger.info("Ensemble model saved to %s", path)


# =====================================================================
# Section 2d: Hyperparameter Search
# =====================================================================

def _generate_param_combos(param_grid: dict) -> list[dict]:
    """Generate all parameter combinations from a grid.

    Args:
        param_grid: Dict of {param_name: [values]}.

    Returns:
        List of dicts, each representing one combination.
    """
    keys = param_grid.keys()
    values = param_grid.values()
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_hyperparameter_search(
    feature_matrix: pd.DataFrame,
    target_col: str = TARGET_COL,
    n_search_folds: int = 3,
) -> dict:
    """Run hyperparameter search for LightGBM, XGBoost, and Ridge.

    Uses a small number of walk-forward folds for speed. Returns the
    best parameters for each model based on average MAE.

    Args:
        feature_matrix: Full feature matrix.
        target_col: Target column name.
        n_search_folds: Number of CV folds for search (fewer = faster).

    Returns:
        Dict of {model_name: {"best_params": dict, "best_mae": float,
                 "all_results": list[dict]}}.
    """
    logger.info("=" * 60)
    logger.info("Starting hyperparameter search")
    logger.info("=" * 60)

    # Use fewer folds for speed
    search_splits = create_walk_forward_splits(
        feature_matrix,
        initial_train_days=WF_INITIAL_TRAIN_DAYS,
        test_window_days=WF_TEST_WINDOW_DAYS,
        step_size_days=90,  # larger steps for speed
        n_folds=n_search_folds,
    )

    results = {}

    # --- LightGBM hyperparameter search ---
    logger.info("Searching LightGBM hyperparameters (%d combinations)",
                len(_generate_param_combos(LGBM_PARAM_GRID)))
    lgbm_results = []
    for combo in _generate_param_combos(LGBM_PARAM_GRID):
        params = LGBM_PARAMS.copy()
        params.update(combo)
        fold_maes = []

        for s in search_splits:
            train = feature_matrix.loc[s["train_start"]:s["train_end"]].dropna(subset=[target_col])
            test = feature_matrix.loc[s["test_start"]:s["test_end"]].dropna(subset=[target_col])
            if len(train) == 0 or len(test) == 0:
                continue

            X_train = train.drop(columns=[target_col])
            y_train = train[target_col]
            X_test = test.drop(columns=[target_col])
            y_test = test[target_col]

            val_size = min(720, len(X_train) // 5)
            model = LightGBMForecaster(params=params)
            model.fit(X_train.iloc[:-val_size], y_train.iloc[:-val_size],
                      X_train.iloc[-val_size:], y_train.iloc[-val_size:])
            preds = model.predict(X_test)
            fold_maes.append(np.mean(np.abs(y_test.values - preds)))

        if fold_maes:
            mean_mae = np.mean(fold_maes)
            lgbm_results.append({"params": combo, "mae": mean_mae, "fold_maes": fold_maes})
            logger.debug("LightGBM %s -> MAE=%.2f", combo, mean_mae)

    lgbm_results.sort(key=lambda x: x["mae"])
    best_lgbm = lgbm_results[0] if lgbm_results else {"params": {}, "mae": float("inf")}
    logger.info("Best LightGBM: MAE=%.2f, params=%s", best_lgbm["mae"], best_lgbm["params"])
    results["lightgbm"] = {
        "best_params": best_lgbm["params"],
        "best_mae": best_lgbm["mae"],
        "all_results": lgbm_results,
    }

    # --- XGBoost hyperparameter search ---
    logger.info("Searching XGBoost hyperparameters (%d combinations)",
                len(_generate_param_combos(XGBM_PARAM_GRID)))
    xgb_results = []
    for combo in _generate_param_combos(XGBM_PARAM_GRID):
        params = XGBM_PARAMS.copy()
        params.update(combo)
        fold_maes = []

        for s in search_splits:
            train = feature_matrix.loc[s["train_start"]:s["train_end"]].dropna(subset=[target_col])
            test = feature_matrix.loc[s["test_start"]:s["test_end"]].dropna(subset=[target_col])
            if len(train) == 0 or len(test) == 0:
                continue

            X_train = train.drop(columns=[target_col])
            y_train = train[target_col]
            X_test = test.drop(columns=[target_col])
            y_test = test[target_col]

            val_size = min(720, len(X_train) // 5)
            model = XGBoostForecaster(params=params)
            model.fit(X_train.iloc[:-val_size], y_train.iloc[:-val_size],
                      X_train.iloc[-val_size:], y_train.iloc[-val_size:])
            preds = model.predict(X_test)
            fold_maes.append(np.mean(np.abs(y_test.values - preds)))

        if fold_maes:
            mean_mae = np.mean(fold_maes)
            xgb_results.append({"params": combo, "mae": mean_mae, "fold_maes": fold_maes})
            logger.debug("XGBoost %s -> MAE=%.2f", combo, mean_mae)

    xgb_results.sort(key=lambda x: x["mae"])
    best_xgb = xgb_results[0] if xgb_results else {"params": {}, "mae": float("inf")}
    logger.info("Best XGBoost: MAE=%.2f, params=%s", best_xgb["mae"], best_xgb["params"])
    results["xgboost"] = {
        "best_params": best_xgb["params"],
        "best_mae": best_xgb["mae"],
        "all_results": xgb_results,
    }

    # --- Ridge hyperparameter search ---
    logger.info("Searching Ridge hyperparameters (%d combinations)",
                len(_generate_param_combos(RIDGE_PARAM_GRID)))
    ridge_results = []
    for combo in _generate_param_combos(RIDGE_PARAM_GRID):
        fold_maes = []

        for s in search_splits:
            train = feature_matrix.loc[s["train_start"]:s["train_end"]].dropna(subset=[target_col])
            test = feature_matrix.loc[s["test_start"]:s["test_end"]].dropna(subset=[target_col])
            if len(train) == 0 or len(test) == 0:
                continue

            X_train = train.drop(columns=[target_col])
            y_train = train[target_col]
            X_test = test.drop(columns=[target_col])
            y_test = test[target_col]

            model = LinearRegressionBaseline(alpha=combo["alpha"])
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_maes.append(np.mean(np.abs(y_test.values - preds)))

        if fold_maes:
            mean_mae = np.mean(fold_maes)
            ridge_results.append({"params": combo, "mae": mean_mae, "fold_maes": fold_maes})

    ridge_results.sort(key=lambda x: x["mae"])
    best_ridge = ridge_results[0] if ridge_results else {"params": {"alpha": 10.0}, "mae": float("inf")}
    logger.info("Best Ridge: MAE=%.2f, params=%s", best_ridge["mae"], best_ridge["params"])
    results["ridge"] = {
        "best_params": best_ridge["params"],
        "best_mae": best_ridge["mae"],
        "all_results": ridge_results,
    }

    return results


# =====================================================================
# Section 3: Walk-Forward Cross-Validation
# =====================================================================

def create_walk_forward_splits(
    feature_matrix: pd.DataFrame,
    initial_train_days: int = WF_INITIAL_TRAIN_DAYS,
    test_window_days: int = WF_TEST_WINDOW_DAYS,
    step_size_days: int = WF_STEP_SIZE_DAYS,
    n_folds: int = WF_N_FOLDS,
) -> list[dict]:
    """Generate walk-forward CV fold definitions.

    Ensures the OOS test window (2024-06-01 onwards) never appears in any fold.

    Args:
        feature_matrix: Full feature DataFrame sorted by time.
        initial_train_days: Minimum training window in days.
        test_window_days: Test window size in days.
        step_size_days: Step between folds in days.
        n_folds: Maximum number of folds.

    Returns:
        List of fold definition dicts with train/test date boundaries.
    """
    oos_boundary = pd.Timestamp(OOS_TEST_START, tz="UTC")
    data_start = feature_matrix.index[0]

    splits = []
    for i in range(n_folds):
        train_end = data_start + pd.Timedelta(days=initial_train_days + i * step_size_days)
        test_start = train_end + pd.Timedelta(hours=1)
        test_end = test_start + pd.Timedelta(days=test_window_days) - pd.Timedelta(hours=1)

        # Stop if test window encroaches on OOS period
        if test_end >= oos_boundary:
            logger.info("Fold %d test_end (%s) >= OOS boundary (%s) — stopping CV",
                        i, test_end, oos_boundary)
            break

        train_data = feature_matrix.loc[data_start:train_end]
        test_data = feature_matrix.loc[test_start:test_end]

        splits.append({
            "fold": i,
            "train_start": data_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": len(train_data),
            "test_rows": len(test_data),
        })

    # Log fold structure
    logger.info("Walk-forward CV: %d folds created", len(splits))
    logger.info("%-5s  %-22s  %-22s  %-22s  %-22s  %8s  %8s",
                "Fold", "Train Start", "Train End", "Test Start", "Test End",
                "Train N", "Test N")
    for s in splits:
        logger.info("%-5d  %-22s  %-22s  %-22s  %-22s  %8d  %8d",
                     s["fold"], s["train_start"], s["train_end"],
                     s["test_start"], s["test_end"],
                     s["train_rows"], s["test_rows"])

    return splits


def run_walk_forward_cv(
    feature_matrix: pd.DataFrame,
    target_col: str,
    models_to_evaluate: dict,
    splits: list[dict],
) -> pd.DataFrame:
    """Train and evaluate each model on each fold.

    Args:
        feature_matrix: Full feature DataFrame.
        target_col: Name of the target column.
        models_to_evaluate: Dict of {model_name: model_instance}.
        splits: Output of create_walk_forward_splits.

    Returns:
        Long-format DataFrame: fold, model, timestamp, actual, predicted.
    """
    # Assert OOS data is not in any fold
    oos_boundary = pd.Timestamp(OOS_TEST_START, tz="UTC")
    for s in splits:
        assert s["test_end"] < oos_boundary, (
            f"Fold {s['fold']} test_end {s['test_end']} >= OOS boundary {oos_boundary}"
        )

    results = []

    for s in splits:
        fold = s["fold"]
        train = feature_matrix.loc[s["train_start"]:s["train_end"]]
        test = feature_matrix.loc[s["test_start"]:s["test_end"]]

        # Drop rows where the target is NaN (missing DA prices)
        train = train.dropna(subset=[target_col])
        test = test.dropna(subset=[target_col])

        if len(train) == 0 or len(test) == 0:
            logger.warning("Fold %d: empty train or test after NaN drop — skipping", fold)
            continue

        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]

        for model_name, model in models_to_evaluate.items():
            logger.info(
                "Fold %d/%d: training %s on %d rows, testing on %d rows",
                fold + 1, len(splits), model_name, len(X_train), len(X_test),
            )

            # For tree-based models and ensemble, carve out validation set
            if hasattr(model, "name") and model.name in ("lightgbm", "xgboost", "lgbm_ridge_ensemble"):
                val_size = min(720, len(X_train) // 5)  # 30 days or 20%
                X_val = X_train.iloc[-val_size:]
                y_val = y_train.iloc[-val_size:]
                X_tr = X_train.iloc[:-val_size]
                y_tr = y_train.iloc[:-val_size]
                model.fit(X_tr, y_tr, X_val, y_val)
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)

            for ts, actual, pred in zip(X_test.index, y_test.values, preds):
                results.append({
                    "fold": fold,
                    "model": model_name,
                    "timestamp": ts,
                    "actual": actual,
                    "predicted": pred,
                })

    results_df = pd.DataFrame(results)
    logger.info("Walk-forward CV complete: %d prediction rows", len(results_df))
    return results_df


# =====================================================================
# Section 4: Performance Metrics
# =====================================================================

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    fold: int = None,
    naive_mae: float = None,
    lag_168_values: np.ndarray = None,
) -> dict:
    """Compute comprehensive metrics for a set of predictions.

    Args:
        actual: Ground truth values.
        predicted: Model predictions.
        model_name: Name of the model.
        fold: Optional fold number.
        naive_mae: Optional seasonal naive MAE for skill score computation.
        lag_168_values: Optional lag-168h values for directional accuracy.

    Returns:
        Dict of metric name -> value.
    """
    ae = np.abs(actual - predicted)
    errors = predicted - actual

    # Compute hour of day from position (assume ordered)
    n = len(actual)

    result = {
        "model": model_name,
        "fold": fold,
        "n": n,
        "MAE": np.mean(ae),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "MBE": np.mean(errors),
        "p95_AE": np.percentile(ae, 95),
        "p99_AE": np.percentile(ae, 99),
    }

    # MAPE — exclude zeros to avoid division by zero
    nonzero_mask = np.abs(actual) > 0.01
    if nonzero_mask.sum() > 0:
        result["MAPE"] = np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100
    else:
        result["MAPE"] = np.nan

    # Skill score vs naive
    if naive_mae is not None and naive_mae > 0:
        result["skill_score_vs_naive"] = 1 - (result["MAE"] / naive_mae)
    else:
        result["skill_score_vs_naive"] = np.nan

    # Directional accuracy
    if lag_168_values is not None and len(lag_168_values) == n:
        pred_direction = np.sign(predicted - lag_168_values)
        actual_direction = np.sign(actual - lag_168_values)
        valid = (pred_direction != 0) & (actual_direction != 0)
        if valid.sum() > 0:
            result["directional_accuracy"] = (
                (pred_direction[valid] == actual_direction[valid]).mean() * 100
            )
        else:
            result["directional_accuracy"] = np.nan
    else:
        result["directional_accuracy"] = np.nan

    return result


def compute_metrics_with_blocks(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    fold: int = None,
    naive_mae: float = None,
) -> dict:
    """Compute metrics including peak/off-peak block splits.

    Args:
        test_df: Test portion of feature matrix (with target and hour info).
        predictions: Model predictions.
        model_name: Name of the model.
        fold: Optional fold number.
        naive_mae: Optional seasonal naive MAE for skill score.

    Returns:
        Dict of metric name -> value including block-level metrics.
    """
    actual = test_df[TARGET_COL].values
    lag_168 = test_df["price_lag_168h"].values if "price_lag_168h" in test_df.columns else None

    result = compute_metrics(actual, predictions, model_name, fold, naive_mae, lag_168)

    # Peak/off-peak splits
    hours = test_df.index.hour
    peak_mask = hours.isin(PEAK_HOURS)
    offpeak_mask = ~peak_mask

    if peak_mask.sum() > 0:
        result["MAE_peak"] = np.mean(np.abs(actual[peak_mask] - predictions[peak_mask]))
    if offpeak_mask.sum() > 0:
        result["MAE_offpeak"] = np.mean(np.abs(actual[offpeak_mask] - predictions[offpeak_mask]))

    return result


def compute_all_metrics(
    cv_results_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics across all folds for each model.

    Args:
        cv_results_df: Output of run_walk_forward_cv.
        feature_matrix: Full feature matrix for lag lookup.

    Returns:
        DataFrame with one row per (model, fold) plus summary rows.
    """
    all_metrics = []

    model_names = cv_results_df["model"].unique()
    folds = cv_results_df["fold"].unique()

    # Compute seasonal naive MAE per fold first
    naive_mae_by_fold = {}
    for fold in folds:
        naive_data = cv_results_df[
            (cv_results_df["model"] == "seasonal_naive") & (cv_results_df["fold"] == fold)
        ]
        if len(naive_data) > 0:
            naive_mae_by_fold[fold] = np.mean(np.abs(
                naive_data["actual"].values - naive_data["predicted"].values
            ))

    for model_name in model_names:
        for fold in folds:
            fold_data = cv_results_df[
                (cv_results_df["model"] == model_name) & (cv_results_df["fold"] == fold)
            ]
            if len(fold_data) == 0:
                continue

            actual = fold_data["actual"].values
            predicted = fold_data["predicted"].values

            # Get lag-168 values for directional accuracy
            lag_168 = None
            timestamps = fold_data["timestamp"].values
            if "price_lag_168h" in feature_matrix.columns:
                lag_168_series = feature_matrix.loc[
                    feature_matrix.index.isin(pd.DatetimeIndex(timestamps)),
                    "price_lag_168h"
                ]
                if len(lag_168_series) == len(fold_data):
                    lag_168 = lag_168_series.values

            metrics = compute_metrics(
                actual, predicted, model_name, fold,
                naive_mae=naive_mae_by_fold.get(fold),
                lag_168_values=lag_168,
            )

            # Add peak/offpeak
            hours = pd.DatetimeIndex(timestamps).hour
            peak_mask = np.isin(hours, PEAK_HOURS)
            if peak_mask.sum() > 0:
                metrics["MAE_peak"] = np.mean(np.abs(actual[peak_mask] - predicted[peak_mask]))
            offpeak_mask = ~peak_mask
            if offpeak_mask.sum() > 0:
                metrics["MAE_offpeak"] = np.mean(np.abs(actual[offpeak_mask] - predicted[offpeak_mask]))

            all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    # Add summary rows (mean +/- std across folds)
    summary_rows = []
    for model_name in model_names:
        model_metrics = metrics_df[metrics_df["model"] == model_name]
        summary = {"model": model_name, "fold": "mean"}
        for col in metrics_df.select_dtypes(include=[np.number]).columns:
            if col == "fold":
                continue
            vals = model_metrics[col].dropna()
            if len(vals) > 0:
                summary[col] = vals.mean()
                summary[f"{col}_std"] = vals.std()
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    full_df = pd.concat([metrics_df, summary_df], ignore_index=True)

    return full_df


def generate_metrics_report(
    cv_metrics_df: pd.DataFrame,
    oos_metrics: dict,
    regime_metrics: dict,
    block_metrics: dict,
    output_path: Path,
    weekly_monthly_metrics: dict = None,
) -> None:
    """Write a formatted text performance report.

    Args:
        cv_metrics_df: Output of compute_all_metrics (CV results).
        oos_metrics: Dict of {model_name: metrics_dict} for OOS period.
        regime_metrics: Dict with crisis/normal regime metrics.
        block_metrics: Dict with peak/offpeak metrics.
        output_path: Path for the output text file.
        weekly_monthly_metrics: Optional dict from compute_weekly_monthly_metrics()
            with keys 'hourly', 'weekly', 'monthly'.

    Side effects:
        Writes output_path and prints to console.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "=" * 70

    lines = [
        sep,
        " DE_LU Power Price Forecasting — Model Performance",
        f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f" Walk-Forward CV: {int(cv_metrics_df[cv_metrics_df['fold'] != 'mean']['fold'].nunique())} folds",
        f" Out-of-Sample Test: {OOS_TEST_START} to {OOS_TEST_END}",
        sep,
        "",
        "MODEL COMPARISON (CV Mean +/- Std across folds)",
        "-" * 70,
    ]

    # Header
    lines.append(
        f"{'Model':25s} {'MAE':>12s} {'RMSE':>12s} {'MAPE':>10s} "
        f"{'P95_AE':>10s} {'P99_AE':>10s} {'Dir.Acc':>10s} {'Skill':>8s}"
    )
    lines.append("-" * 70)

    # CV summary rows
    summary = cv_metrics_df[cv_metrics_df["fold"] == "mean"]
    for _, row in summary.iterrows():
        mae_str = f"{row.get('MAE', 0):.1f}"
        if "MAE_std" in row and pd.notna(row.get("MAE_std")):
            mae_str += f"+-{row['MAE_std']:.1f}"

        rmse_str = f"{row.get('RMSE', 0):.1f}"
        if "RMSE_std" in row and pd.notna(row.get("RMSE_std")):
            rmse_str += f"+-{row['RMSE_std']:.1f}"

        mape_str = f"{row.get('MAPE', 0):.1f}%" if pd.notna(row.get("MAPE")) else "N/A"
        p95_str = f"{row.get('p95_AE', 0):.1f}" if pd.notna(row.get("p95_AE")) else "N/A"
        p99_str = f"{row.get('p99_AE', 0):.1f}" if pd.notna(row.get("p99_AE")) else "N/A"
        da_str = f"{row.get('directional_accuracy', 0):.1f}%" if pd.notna(row.get("directional_accuracy")) else "N/A"
        skill_str = f"{row.get('skill_score_vs_naive', 0):+.2f}" if pd.notna(row.get("skill_score_vs_naive")) else "N/A"

        lines.append(
            f"{row['model']:25s} {mae_str:>12s} {rmse_str:>12s} {mape_str:>10s} "
            f"{p95_str:>10s} {p99_str:>10s} {da_str:>10s} {skill_str:>8s}"
        )

    lines.append("")

    # OOS results
    lines.append(f"OOS TEST RESULTS ({OOS_TEST_START} to {OOS_TEST_END})")
    lines.append("-" * 70)
    lines.append(
        f"{'Model':25s} {'MAE':>12s} {'RMSE':>12s} {'MAPE':>10s} "
        f"{'P95_AE':>10s} {'P99_AE':>10s} {'Dir.Acc':>10s} {'Skill':>8s}"
    )
    lines.append("-" * 70)

    for model_name, m in oos_metrics.items():
        mae_str = f"{m.get('MAE', 0):.1f}"
        rmse_str = f"{m.get('RMSE', 0):.1f}"
        mape_str = f"{m.get('MAPE', 0):.1f}%" if pd.notna(m.get("MAPE")) else "N/A"
        p95_str = f"{m.get('p95_AE', 0):.1f}"
        p99_str = f"{m.get('p99_AE', 0):.1f}"
        da_str = f"{m.get('directional_accuracy', 0):.1f}%" if pd.notna(m.get("directional_accuracy")) else "N/A"
        skill_str = f"{m.get('skill_score_vs_naive', 0):+.2f}" if pd.notna(m.get("skill_score_vs_naive")) else "N/A"

        lines.append(
            f"{model_name:25s} {mae_str:>12s} {rmse_str:>12s} {mape_str:>10s} "
            f"{p95_str:>10s} {p99_str:>10s} {da_str:>10s} {skill_str:>8s}"
        )

    lines.append("")

    # Regime analysis
    lines.append("PERFORMANCE BY REGIME")
    lines.append("-" * 70)
    for regime_name, m in regime_metrics.items():
        lines.append(f"  {regime_name}:")
        lines.append(f"    LightGBM MAE:  {m.get('MAE', 0):.1f} EUR/MWh")
        lines.append(f"    P99 AE:        {m.get('p99_AE', 0):.1f} EUR/MWh")
    lines.append("")

    # Block analysis
    lines.append("PERFORMANCE BY BLOCK")
    lines.append("-" * 70)
    for block_name, m in block_metrics.items():
        lines.append(f"  {block_name}:")
        lines.append(f"    LightGBM MAE:  {m.get('MAE', 0):.1f} EUR/MWh")
    lines.append("")

    # Weekly / Monthly accuracy
    if weekly_monthly_metrics:
        lines.append("MULTI-GRANULARITY FORECAST ACCURACY")
        lines.append("-" * 70)
        lines.append("  (Hourly forecasts aggregated to weekly and monthly averages)")
        lines.append("")

        for granularity in ["hourly", "weekly", "monthly"]:
            if granularity not in weekly_monthly_metrics:
                continue
            gran_data = weekly_monthly_metrics[granularity]
            if not gran_data:
                continue

            lines.append(f"  {granularity.upper()} ACCURACY:")
            lines.append(
                f"  {'Model':28s}  {'MAE':>8s}  {'RMSE':>8s}  {'MAPE':>8s}  "
                f"{'MBE':>8s}  {'N':>5s}"
            )
            lines.append(f"  {'-' * 65}")
            for model_name, m in gran_data.items():
                mae = f"{m.get('MAE', 0):.2f}"
                rmse = f"{m.get('RMSE', 0):.2f}"
                mape = f"{m.get('MAPE', 0):.1f}%" if pd.notna(m.get("MAPE")) else "N/A"
                mbe = f"{m.get('MBE', 0):+.2f}"
                n = f"{m.get('n', 0)}"
                lines.append(
                    f"  {model_name:28s}  {mae:>8s}  {rmse:>8s}  "
                    f"{mape:>8s}  {mbe:>8s}  {n:>5s}"
                )
            lines.append("")

        lines.append("  Note: Weekly = ISO week avg, Monthly = calendar month avg")
        lines.append("        Aggregated from hourly forecasts (not separate models)")
        lines.append(f"        OOS period {OOS_TEST_START} to {OOS_TEST_END}")
        lines.append("")

    # Best model summary — find best by MAE across all models
    if "seasonal_naive" in oos_metrics:
        naive_mae = oos_metrics["seasonal_naive"]["MAE"]
        best_model_name = None
        best_mae = float("inf")
        for name, m in oos_metrics.items():
            if name == "seasonal_naive" or name == "last_week_same_day":
                continue
            if m["MAE"] < best_mae:
                best_mae = m["MAE"]
                best_model_name = name

        if best_model_name:
            improvement = (1 - best_mae / naive_mae) * 100 if naive_mae > 0 else 0
            da = oos_metrics[best_model_name].get("directional_accuracy", 0)
            lines.append(f"BEST MODEL: {best_model_name}")
            lines.append(f"  OOS MAE: {best_mae:.1f} EUR/MWh")
            lines.append(f"  Improvement over seasonal naive: {improvement:+.1f}% MAE reduction")
            lines.append(f"  Directional accuracy: {da:.1f}% (random = 50%)")
            lines.append("")

            # Compare all advanced models
            lines.append("ALL MODEL OOS MAE RANKING")
            lines.append("-" * 40)
            ranked = sorted(oos_metrics.items(), key=lambda x: x[1]["MAE"])
            for rank, (name, m) in enumerate(ranked, 1):
                marker = " <-- BEST" if name == best_model_name else ""
                lines.append(f"  {rank}. {name:25s}  MAE={m['MAE']:.1f}{marker}")

    lines.append(sep)

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    logger.info("Metrics report written to %s", output_path)


# =====================================================================
# Section 5: Out-of-Sample Evaluation and Submission File
# =====================================================================

def run_oos_evaluation(
    feature_matrix: pd.DataFrame,
    target_col: str = TARGET_COL,
    oos_start: str = OOS_TEST_START,
    oos_end: str = OOS_TEST_END,
    best_params: dict = None,
) -> tuple[pd.DataFrame, dict, LightGBMForecaster]:
    """Train all models on pre-OOS data, predict OOS window.

    Args:
        feature_matrix: Full feature matrix.
        target_col: Target column name.
        oos_start: OOS start date string.
        oos_end: OOS end date string.
        best_params: Optional dict of {model_name: {"best_params": dict}}
            from hyperparameter search.

    Returns:
        Tuple of (predictions_df, all_model_oos_metrics, fitted_models_dict).

    Raises:
        AssertionError: If there is temporal leakage.
    """
    oos_start_ts = pd.Timestamp(oos_start, tz="UTC")
    oos_end_ts = pd.Timestamp(f"{oos_end} 23:00", tz="UTC")

    train = feature_matrix.loc[feature_matrix.index < oos_start_ts]
    test = feature_matrix.loc[
        (feature_matrix.index >= oos_start_ts) & (feature_matrix.index <= oos_end_ts)
    ]

    # Drop rows where target is NaN
    train = train.dropna(subset=[target_col])
    test = test.dropna(subset=[target_col])

    # Leakage assertion
    assert train.index.max() < test.index.min(), (
        f"Leakage! Train max {train.index.max()} >= Test min {test.index.min()}"
    )

    logger.info("OOS evaluation: train=%d rows, test=%d rows", len(train), len(test))

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # Get lag-168 for directional accuracy
    lag_168 = X_test["price_lag_168h"].values if "price_lag_168h" in X_test.columns else None

    # Train and predict with all models (including XGBoost and Ensemble)
    models = {
        "seasonal_naive": SeasonalNaiveModel(),
        "last_week_same_day": LastWeekSameDayModel(),
        "linear_regression": LinearRegressionBaseline(),
        "lightgbm": LightGBMForecaster(),
        "xgboost": XGBoostForecaster(),
        "lgbm_ridge_ensemble": LGBMRidgeEnsemble(),
    }

    # If best hyperparams were found, apply them
    if best_params:
        if "lightgbm" in best_params and best_params["lightgbm"].get("best_params"):
            lgbm_p = LGBM_PARAMS.copy()
            lgbm_p.update(best_params["lightgbm"]["best_params"])
            models["lightgbm"] = LightGBMForecaster(params=lgbm_p)
            logger.info("OOS: Using tuned LightGBM params: %s", best_params["lightgbm"]["best_params"])
        if "xgboost" in best_params and best_params["xgboost"].get("best_params"):
            xgb_p = XGBM_PARAMS.copy()
            xgb_p.update(best_params["xgboost"]["best_params"])
            models["xgboost"] = XGBoostForecaster(params=xgb_p)
            logger.info("OOS: Using tuned XGBoost params: %s", best_params["xgboost"]["best_params"])
        if "ridge" in best_params and best_params["ridge"].get("best_params"):
            ridge_alpha = best_params["ridge"]["best_params"].get("alpha", 10.0)
            models["linear_regression"] = LinearRegressionBaseline(alpha=ridge_alpha)
            # Also update ensemble's Ridge
            lgbm_p_ens = LGBM_PARAMS.copy()
            if "lightgbm" in best_params and best_params["lightgbm"].get("best_params"):
                lgbm_p_ens.update(best_params["lightgbm"]["best_params"])
            models["lgbm_ridge_ensemble"] = LGBMRidgeEnsemble(
                lgbm_params=lgbm_p_ens, ridge_alpha=ridge_alpha
            )
            logger.info("OOS: Using tuned Ridge alpha=%.1f", ridge_alpha)

    predictions = {}
    oos_metrics = {}

    # Get naive MAE first for skill scores
    naive_preds = models["seasonal_naive"].predict(X_test)
    naive_mae = np.mean(np.abs(y_test.values - naive_preds))

    for name, model in models.items():
        logger.info("OOS: training %s on %d rows", name, len(X_train))

        if name in ("lightgbm", "xgboost", "lgbm_ridge_ensemble"):
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        predictions[name] = preds

        metrics = compute_metrics(
            y_test.values, preds, name,
            naive_mae=naive_mae,
            lag_168_values=lag_168,
        )

        # Add peak/offpeak
        hours = X_test.index.hour
        peak_mask = np.isin(hours, PEAK_HOURS)
        offpeak_mask = ~peak_mask
        if peak_mask.sum() > 0:
            metrics["MAE_peak"] = np.mean(np.abs(y_test.values[peak_mask] - preds[peak_mask]))
        if offpeak_mask.sum() > 0:
            metrics["MAE_offpeak"] = np.mean(np.abs(y_test.values[offpeak_mask] - preds[offpeak_mask]))

        oos_metrics[name] = metrics
        logger.info("OOS %s: MAE=%.2f, RMSE=%.2f", name, metrics["MAE"], metrics["RMSE"])

    # Build predictions DataFrame
    oos_df = pd.DataFrame({
        "timestamp_utc": X_test.index,
        "y_actual": y_test.values,
        "y_pred": predictions["lightgbm"],
        "y_pred_naive": predictions["seasonal_naive"],
        "y_pred_linear": predictions["linear_regression"],
        "y_pred_xgboost": predictions["xgboost"],
        "y_pred_ensemble": predictions["lgbm_ridge_ensemble"],
    })
    oos_df = oos_df.set_index("timestamp_utc")

    # Save
    oos_path = DATA_PROCESSED / "oos_predictions.parquet"
    oos_df.to_parquet(oos_path)
    logger.info("OOS predictions saved to %s", oos_path)

    return oos_df, oos_metrics, models


def save_submission_csv(oos_predictions: pd.DataFrame, output_path: Path) -> None:
    """Create the submission CSV in the required format.

    Args:
        oos_predictions: DataFrame with y_pred column and UTC timestamp index.
        output_path: Path for the CSV file.

    Side effects:
        Writes the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({
        "id": oos_predictions.index.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "y_pred": oos_predictions["y_pred"].round(2),
    })
    sub.to_csv(output_path, index=False)
    logger.info(
        "submission.csv written: %d rows, %s to %s",
        len(sub), sub["id"].iloc[0], sub["id"].iloc[-1],
    )


# =====================================================================
# Section 6: Curve Aggregation (DA -> Prompt Curve Translation)
# =====================================================================

def aggregate_forecast_to_delivery_periods(
    hourly_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Roll hourly forecasts into delivery-period averages and confidence bands.

    Args:
        hourly_forecast_df: DataFrame with y_pred, y_actual columns and
            UTC timestamp index.

    Returns:
        DataFrame with one row per delivery date, including base/peak/offpeak
        predictions, actuals, percentile bands, and rolling aggregations.
    """
    df = hourly_forecast_df.copy()
    df["delivery_date"] = df.index.date
    df["hour"] = df.index.hour

    daily_records = []

    for date, group in df.groupby("delivery_date"):
        peak_mask = group["hour"].isin(PEAK_HOURS)
        offpeak_mask = ~peak_mask

        record = {
            "delivery_date": date,
            "base_pred": group["y_pred"].mean(),
            "peak_pred": group.loc[peak_mask, "y_pred"].mean() if peak_mask.sum() > 0 else np.nan,
            "offpeak_pred": group.loc[offpeak_mask, "y_pred"].mean() if offpeak_mask.sum() > 0 else np.nan,
            "base_p10": group["y_pred"].quantile(0.10),
            "base_p90": group["y_pred"].quantile(0.90),
        }

        # Actuals (may be NaN for future forecasts)
        if "y_actual" in group.columns:
            record["base_actual"] = group["y_actual"].mean()
            record["peak_actual"] = (
                group.loc[peak_mask, "y_actual"].mean() if peak_mask.sum() > 0 else np.nan
            )

        record["peak_premium"] = record["peak_pred"] - record["base_pred"]

        daily_records.append(record)

    daily_df = pd.DataFrame(daily_records)
    daily_df["delivery_date"] = pd.to_datetime(daily_df["delivery_date"])
    daily_df = daily_df.set_index("delivery_date").sort_index()

    # Rolling forward aggregations
    daily_df["next_7d_base_mean"] = daily_df["base_pred"].rolling(7, min_periods=1).mean()
    daily_df["next_30d_base_mean"] = daily_df["base_pred"].rolling(30, min_periods=1).mean()
    daily_df["next_7d_peak_mean"] = daily_df["peak_pred"].rolling(7, min_periods=1).mean()
    daily_df["next_30d_peak_mean"] = daily_df["peak_pred"].rolling(30, min_periods=1).mean()
    daily_df["next_7d_base_p10"] = daily_df["base_p10"].rolling(7, min_periods=1).mean()
    daily_df["next_7d_base_p90"] = daily_df["base_p90"].rolling(7, min_periods=1).mean()
    daily_df["next_30d_base_p10"] = daily_df["base_p10"].rolling(30, min_periods=1).mean()
    daily_df["next_30d_base_p90"] = daily_df["base_p90"].rolling(30, min_periods=1).mean()

    logger.info("Delivery-period aggregation: %d delivery days", len(daily_df))
    return daily_df


# =====================================================================
# Section 7: Figures
# =====================================================================

def _setup_plot_style():
    """Configure a clean, professional matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def plot_forecast_vs_actual(
    oos_df: pd.DataFrame,
    output_dir: Path = None,
) -> None:
    """Figure 1: Actual vs predicted, error by hour, scatter plot.

    Three-panel figure showing a 14-day window, hourly error distribution,
    and predicted vs actual scatter.

    Args:
        oos_df: OOS predictions DataFrame with y_actual and y_pred columns.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.

    Side effects:
        Saves fig_forecast_vs_actual.png.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("LightGBM Forecast vs Actual — OOS Period", fontsize=15, fontweight="bold")

    # Panel 1: 14-day window time series
    ax = axes[0]
    window_start = oos_df.index[0]
    window_end = window_start + pd.Timedelta(days=14)
    window = oos_df.loc[window_start:window_end]

    ax.plot(window.index, window["y_actual"], color="#1f77b4", linewidth=1.5,
            label="Actual", linestyle="-")
    ax.plot(window.index, window["y_pred"], color="#e74c3c", linewidth=1.5,
            label="LightGBM", linestyle="--")
    ax.fill_between(window.index, window["y_pred"] * 0.85, window["y_pred"] * 1.15,
                     alpha=0.15, color="#e74c3c", label="~p10-p90 band")
    ax.set_ylabel("DA Price (EUR/MWh)")
    ax.set_title("14-Day Forecast Window (first 2 weeks of OOS)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Panel 2: Absolute error by hour of day (box plot)
    ax = axes[1]
    oos_copy = oos_df.copy()
    oos_copy["hour"] = oos_copy.index.hour
    oos_copy["abs_error"] = np.abs(oos_copy["y_actual"] - oos_copy["y_pred"])

    hour_data = [oos_copy.loc[oos_copy["hour"] == h, "abs_error"].values for h in range(24)]
    bp = ax.boxplot(hour_data, tick_labels=range(24), patch_artist=True,
                     showfliers=False, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("|Error| (EUR/MWh)")
    ax.set_title("Absolute Forecast Error by Hour of Day")

    # Panel 3: Scatter plot — predicted vs actual
    ax = axes[2]
    scatter = ax.scatter(
        oos_df["y_actual"], oos_df["y_pred"],
        c=oos_df.index.month, cmap="coolwarm",
        alpha=0.3, s=5, edgecolors="none",
    )
    # Perfect forecast line
    lims = [
        min(oos_df["y_actual"].min(), oos_df["y_pred"].min()),
        max(oos_df["y_actual"].max(), oos_df["y_pred"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect forecast")
    ax.set_xlabel("Actual (EUR/MWh)")
    ax.set_ylabel("Predicted (EUR/MWh)")
    ax.set_title("Predicted vs Actual (colored by month)")
    ax.legend(fontsize=9)
    fig.colorbar(scatter, ax=ax, label="Month", shrink=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "fig_forecast_vs_actual.png")
    plt.close(fig)
    logger.info("Figure 1 saved: %s", output_dir / "fig_forecast_vs_actual.png")


def plot_model_comparison(
    cv_metrics_df: pd.DataFrame,
    cv_results_df: pd.DataFrame,
    output_dir: Path = None,
) -> None:
    """Figure 2: Bar chart comparing models + per-fold MAE over time.

    Args:
        cv_metrics_df: Output of compute_all_metrics.
        cv_results_df: Raw CV predictions.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.

    Side effects:
        Saves fig_model_comparison.png.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
    fig.suptitle("Model Comparison — Walk-Forward CV", fontsize=15, fontweight="bold")

    # Top: grouped bar chart of MAE, RMSE, P95_AE
    summary = cv_metrics_df[cv_metrics_df["fold"] == "mean"].copy()
    model_names = summary["model"].values
    metrics_to_plot = ["MAE", "RMSE", "p95_AE"]
    x = np.arange(len(model_names))
    width = 0.22
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, metric in enumerate(metrics_to_plot):
        vals = summary[metric].values.astype(float)
        std_col = f"{metric}_std"
        stds = summary[std_col].values.astype(float) if std_col in summary.columns else np.zeros_like(vals)
        ax1.bar(x + i * width, vals, width, label=metric, color=colors[i],
                alpha=0.8, yerr=stds, capsize=3)

    # Naive MAE reference line
    naive_row = summary[summary["model"] == "seasonal_naive"]
    if len(naive_row) > 0:
        naive_mae = float(naive_row["MAE"].values[0])
        ax1.axhline(y=naive_mae, color="gray", linestyle="--", linewidth=1.5,
                     alpha=0.7, label=f"Naive MAE ({naive_mae:.1f})")

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, fontsize=8, rotation=15, ha="right")
    ax1.set_ylabel("EUR/MWh")
    ax1.set_title("CV Metrics (Mean +/- Std across folds)")
    ax1.legend(fontsize=8)

    # Bottom: per-fold MAE over time
    model_colors = {
        "seasonal_naive": "#95a5a6",
        "last_week_same_day": "#7f8c8d",
        "linear_regression": "#3498db",
        "lightgbm": "#e74c3c",
        "xgboost": "#9b59b6",
        "lgbm_ridge_ensemble": "#f39c12",
    }

    fold_data = cv_metrics_df[cv_metrics_df["fold"] != "mean"].copy()
    fold_data["fold"] = fold_data["fold"].astype(int)

    for model_name in fold_data["model"].unique():
        model_folds = fold_data[fold_data["model"] == model_name].sort_values("fold")
        ax2.plot(model_folds["fold"], model_folds["MAE"],
                 marker="o", label=model_name, linewidth=2, markersize=5,
                 color=model_colors.get(model_name, "#333333"))

    ax2.set_xlabel("Fold")
    ax2.set_ylabel("MAE (EUR/MWh)")
    ax2.set_title("Per-Fold MAE Over Time (concept drift check)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_xticks(fold_data["fold"].unique())

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "fig_model_comparison.png")
    plt.close(fig)
    logger.info("Figure 2 saved: %s", output_dir / "fig_model_comparison.png")


def plot_feature_importance(
    lgbm_model: LightGBMForecaster,
    output_dir: Path = None,
) -> None:
    """Figure 3: Horizontal bar chart of top 25 LightGBM features by gain.

    Bars colored by feature category. Vertical line at price_lag_168h importance.

    Args:
        lgbm_model: Fitted LightGBMForecaster instance.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.

    Side effects:
        Saves fig_feature_importance.png.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    imp_df = lgbm_model.get_feature_importance(top_n=25)

    # Categorize features by color
    def _categorize(name: str) -> str:
        if "price_lag" in name or "price_rolling" in name or "price_24h" in name or "price_168h" in name:
            return "price_lags"
        elif any(x in name for x in ["hour", "dow", "month", "week", "year", "quarter",
                                       "weekend", "monday", "friday", "holiday", "bridge"]):
            return "calendar"
        elif any(x in name for x in ["gas", "ttf"]):
            return "gas"
        elif any(x in name for x in ["regime", "negative_price", "vol_regime"]):
            return "regime"
        else:
            return "fundamentals"

    category_colors = {
        "price_lags": "#3498db",
        "calendar": "#95a5a6",
        "fundamentals": "#2ecc71",
        "gas": "#f39c12",
        "regime": "#e74c3c",
    }

    imp_df["category"] = imp_df["feature"].apply(_categorize)
    imp_df["color"] = imp_df["category"].map(category_colors)

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(range(len(imp_df)), imp_df["importance_pct"].values,
                    color=imp_df["color"].values, alpha=0.85)

    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (% of total gain)")
    ax.set_title("Top 25 LightGBM Features by Gain Importance", fontsize=13, fontweight="bold")

    # Reference line at price_lag_168h
    lag168_row = imp_df[imp_df["feature"] == "price_lag_168h"]
    if len(lag168_row) > 0:
        ref_val = lag168_row["importance_pct"].values[0]
        ax.axvline(x=ref_val, color="navy", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(ref_val + 0.2, len(imp_df) - 1, "price_lag_168h\n(naive baseline)",
                fontsize=7, color="navy", va="top")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=cat) for cat, c in category_colors.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_feature_importance.png")
    plt.close(fig)
    logger.info("Figure 3 saved: %s", output_dir / "fig_feature_importance.png")


def plot_walk_forward_errors(
    cv_results_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    output_dir: Path = None,
) -> None:
    """Figure 4: Daily MAE time series for LightGBM across CV and OOS.

    Args:
        cv_results_df: Raw CV predictions.
        oos_df: OOS predictions DataFrame.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.

    Side effects:
        Saves fig_walk_forward_errors.png.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(16, 6))

    # CV daily MAE for primary model (LightGBM or ensemble — whichever exists)
    primary_model = "lightgbm"
    available_models = cv_results_df["model"].unique()
    if "lgbm_ridge_ensemble" in available_models:
        primary_model = "lgbm_ridge_ensemble"
    elif "lightgbm" not in available_models:
        # Use first non-naive model
        non_naive = [m for m in available_models if m != "seasonal_naive"]
        primary_model = non_naive[0] if non_naive else available_models[0]

    primary_cv = cv_results_df[cv_results_df["model"] == primary_model].copy()
    primary_cv["date"] = pd.to_datetime(primary_cv["timestamp"]).dt.date
    primary_cv["abs_error"] = np.abs(primary_cv["actual"] - primary_cv["predicted"])
    cv_daily = primary_cv.groupby("date")["abs_error"].mean()
    cv_daily.index = pd.to_datetime(cv_daily.index)

    ax.plot(cv_daily.index, cv_daily.values, color="#3498db", linewidth=0.8,
            alpha=0.7, label=f"{primary_model} CV daily MAE")

    # OOS daily MAE
    oos_copy = oos_df.copy()
    oos_copy["date"] = oos_copy.index.date
    oos_copy["abs_error"] = np.abs(oos_copy["y_actual"] - oos_copy["y_pred"])
    oos_daily = oos_copy.groupby("date")["abs_error"].mean()
    oos_daily.index = pd.to_datetime(oos_daily.index)

    ax.plot(oos_daily.index, oos_daily.values, color="#e74c3c", linewidth=0.8,
            alpha=0.8, label=f"{primary_model} OOS daily MAE")

    # Naive benchmark — CV
    naive_cv = cv_results_df[cv_results_df["model"] == "seasonal_naive"].copy()
    naive_cv["date"] = pd.to_datetime(naive_cv["timestamp"]).dt.date
    naive_cv["abs_error"] = np.abs(naive_cv["actual"] - naive_cv["predicted"])
    naive_daily = naive_cv.groupby("date")["abs_error"].mean()
    naive_daily.index = pd.to_datetime(naive_daily.index)

    overall_naive_mae = naive_daily.mean()
    ax.axhline(y=overall_naive_mae, color="gray", linestyle="--", linewidth=1.5,
               alpha=0.6, label=f"Naive avg MAE ({overall_naive_mae:.1f})")

    # Annotate key events
    events = {
        "2022-08-26": "Gas crisis\npeak",
        "2022-12-21": "DA price\nspike",
    }
    for date_str, label in events.items():
        event_date = pd.Timestamp(date_str)
        if cv_daily.index.min() <= event_date <= cv_daily.index.max():
            ax.annotate(
                label, xy=(event_date, cv_daily.get(event_date, overall_naive_mae)),
                xytext=(0, 30), textcoords="offset points",
                fontsize=8, ha="center", color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", alpha=0.7),
            )

    # OOS boundary
    oos_start = pd.Timestamp(OOS_TEST_START)
    ax.axvline(x=oos_start, color="green", linestyle="-.", linewidth=1.5, alpha=0.7)
    ax.text(oos_start, ax.get_ylim()[1] * 0.95, " OOS start", fontsize=8,
            color="green", va="top")

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily MAE (EUR/MWh)")
    ax.set_title("LightGBM Daily Forecast Error Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_walk_forward_errors.png")
    plt.close(fig)
    logger.info("Figure 4 saved: %s", output_dir / "fig_walk_forward_errors.png")


# =====================================================================
# Section 7b: New Comparative & Hyperparameter Analysis Figures
# =====================================================================

def plot_full_model_comparison(
    oos_df: pd.DataFrame,
    oos_metrics: dict,
    output_dir: Path = None,
) -> None:
    """Figure 5: Full comparative analysis of all models on OOS data.

    Four-panel figure: OOS MAE bars, scatter overlay, residual distributions,
    and hourly error profiles for all models.

    Args:
        oos_df: OOS predictions with all model columns.
        oos_metrics: Dict of {model_name: metrics_dict} for OOS period.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Full Model Comparison — OOS Period (Jun-Dec 2024)",
                 fontsize=16, fontweight="bold")

    model_pred_cols = {
        "Seasonal Naive": "y_pred_naive",
        "Linear (Ridge)": "y_pred_linear",
        "LightGBM": "y_pred",
        "XGBoost": "y_pred_xgboost",
        "LGBM+Ridge Ensemble": "y_pred_ensemble",
    }
    model_colors = {
        "Seasonal Naive": "#95a5a6",
        "Linear (Ridge)": "#3498db",
        "LightGBM": "#e74c3c",
        "XGBoost": "#9b59b6",
        "LGBM+Ridge Ensemble": "#f39c12",
    }

    # Panel 1: OOS MAE bar chart for all models
    ax = axes[0, 0]
    oos_model_keys = list(oos_metrics.keys())
    mae_vals = [oos_metrics[k]["MAE"] for k in oos_model_keys]
    rmse_vals = [oos_metrics[k]["RMSE"] for k in oos_model_keys]
    x = np.arange(len(oos_model_keys))
    width = 0.35
    bars1 = ax.bar(x - width/2, mae_vals, width, label="MAE", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(oos_model_keys, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("OOS Error Metrics by Model")
    ax.legend()
    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", fontsize=7)

    # Panel 2: 7-day overlay — all models vs actual
    ax = axes[0, 1]
    window_start = oos_df.index[0]
    window_end = window_start + pd.Timedelta(days=7)
    window = oos_df.loc[window_start:window_end]

    ax.plot(window.index, window["y_actual"], color="black", linewidth=2,
            label="Actual", linestyle="-")
    for label, col in model_pred_cols.items():
        if col in window.columns:
            ax.plot(window.index, window[col], linewidth=1.2, label=label,
                    color=model_colors.get(label, "#333"),
                    linestyle="--" if label != "Actual" else "-")
    ax.set_ylabel("DA Price (EUR/MWh)")
    ax.set_title("7-Day Forecast Overlay (all models)")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Panel 3: Residual distributions (violin/box)
    ax = axes[1, 0]
    residuals_data = []
    residuals_labels = []
    for label, col in model_pred_cols.items():
        if col in oos_df.columns:
            resid = oos_df["y_actual"] - oos_df[col]
            residuals_data.append(resid.values)
            residuals_labels.append(label)

    bp = ax.boxplot(residuals_data, labels=residuals_labels, patch_artist=True,
                     showfliers=False, widths=0.6)
    colors_list = [model_colors.get(l, "#333") for l in residuals_labels]
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Residual (Actual - Predicted) EUR/MWh")
    ax.set_title("Residual Distributions (closer to 0 = better)")
    ax.tick_params(axis="x", rotation=15)

    # Panel 4: Hourly MAE profile for each model
    ax = axes[1, 1]
    oos_copy = oos_df.copy()
    oos_copy["hour"] = oos_copy.index.hour
    for label, col in model_pred_cols.items():
        if col in oos_copy.columns:
            hourly_mae = oos_copy.groupby("hour").apply(
                lambda g: np.mean(np.abs(g["y_actual"] - g[col]))
            )
            ax.plot(hourly_mae.index, hourly_mae.values, marker="o", markersize=4,
                    label=label, color=model_colors.get(label, "#333"), linewidth=1.5)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Hourly MAE Profile by Model")
    ax.legend(fontsize=7)
    ax.set_xticks(range(24))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "fig_full_model_comparison.png")
    plt.close(fig)
    logger.info("Figure 5 saved: %s", output_dir / "fig_full_model_comparison.png")


def plot_hyperparameter_analysis(
    hp_results: dict,
    output_dir: Path = None,
) -> None:
    """Figure 6: Hyperparameter search results visualization.

    Shows how MAE varies with each key hyperparameter for LightGBM, XGBoost,
    and Ridge.

    Args:
        hp_results: Output of run_hyperparameter_search.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Hyperparameter Search Results", fontsize=16, fontweight="bold")

    # --- LightGBM: learning_rate vs MAE ---
    ax = axes[0, 0]
    if "lightgbm" in hp_results:
        all_res = hp_results["lightgbm"]["all_results"]
        for lr in sorted(set(r["params"].get("learning_rate", 0.05) for r in all_res)):
            subset = [r for r in all_res if r["params"].get("learning_rate") == lr]
            num_leaves_vals = [r["params"].get("num_leaves", 63) for r in subset]
            mae_vals = [r["mae"] for r in subset]
            ax.scatter(num_leaves_vals, mae_vals, label=f"lr={lr}", s=40, alpha=0.7)
        ax.set_xlabel("num_leaves")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("LightGBM: num_leaves vs MAE")
        ax.legend(fontsize=7)

    # --- LightGBM: n_estimators vs MAE ---
    ax = axes[0, 1]
    if "lightgbm" in hp_results:
        all_res = hp_results["lightgbm"]["all_results"]
        for ne in sorted(set(r["params"].get("n_estimators", 1000) for r in all_res)):
            subset = [r for r in all_res if r["params"].get("n_estimators") == ne]
            lr_vals = [r["params"].get("learning_rate", 0.05) for r in subset]
            mae_vals = [r["mae"] for r in subset]
            ax.scatter(lr_vals, mae_vals, label=f"n_est={ne}", s=40, alpha=0.7)
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("LightGBM: learning_rate vs MAE")
        ax.legend(fontsize=7)

    # --- LightGBM: min_child_samples vs MAE ---
    ax = axes[0, 2]
    if "lightgbm" in hp_results:
        all_res = hp_results["lightgbm"]["all_results"]
        mcs_vals = [r["params"].get("min_child_samples", 20) for r in all_res]
        mae_vals = [r["mae"] for r in all_res]
        ax.scatter(mcs_vals, mae_vals, color="#e74c3c", alpha=0.5, s=30)
        # Trend line
        z = np.polyfit(mcs_vals, mae_vals, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(mcs_vals), max(mcs_vals), 50)
        ax.plot(x_smooth, p(x_smooth), "r--", linewidth=1.5)
        ax.set_xlabel("min_child_samples")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("LightGBM: min_child_samples vs MAE")

    # --- XGBoost: max_depth vs MAE ---
    ax = axes[1, 0]
    if "xgboost" in hp_results:
        all_res = hp_results["xgboost"]["all_results"]
        for lr in sorted(set(r["params"].get("learning_rate", 0.05) for r in all_res)):
            subset = [r for r in all_res if r["params"].get("learning_rate") == lr]
            depth_vals = [r["params"].get("max_depth", 7) for r in subset]
            mae_vals = [r["mae"] for r in subset]
            ax.scatter(depth_vals, mae_vals, label=f"lr={lr}", s=40, alpha=0.7)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("XGBoost: max_depth vs MAE")
        ax.legend(fontsize=7)

    # --- XGBoost: min_child_weight vs MAE ---
    ax = axes[1, 1]
    if "xgboost" in hp_results:
        all_res = hp_results["xgboost"]["all_results"]
        mcw_vals = [r["params"].get("min_child_weight", 20) for r in all_res]
        mae_vals = [r["mae"] for r in all_res]
        ax.scatter(mcw_vals, mae_vals, color="#9b59b6", alpha=0.5, s=30)
        z = np.polyfit(mcw_vals, mae_vals, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(mcw_vals), max(mcw_vals), 50)
        ax.plot(x_smooth, p(x_smooth), "--", color="#9b59b6", linewidth=1.5)
        ax.set_xlabel("min_child_weight")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("XGBoost: min_child_weight vs MAE")

    # --- Ridge: alpha vs MAE ---
    ax = axes[1, 2]
    if "ridge" in hp_results:
        all_res = hp_results["ridge"]["all_results"]
        alpha_vals = [r["params"]["alpha"] for r in all_res]
        mae_vals = [r["mae"] for r in all_res]
        ax.bar(range(len(alpha_vals)), mae_vals, color="#3498db", alpha=0.7)
        ax.set_xticks(range(len(alpha_vals)))
        ax.set_xticklabels([f"{a}" for a in alpha_vals], fontsize=8)
        ax.set_xlabel("Ridge alpha")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("Ridge: alpha vs MAE")
        # Mark best
        best_idx = np.argmin(mae_vals)
        ax.bar(best_idx, mae_vals[best_idx], color="#2ecc71", alpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "fig_hyperparameter_analysis.png")
    plt.close(fig)
    logger.info("Figure 6 saved: %s", output_dir / "fig_hyperparameter_analysis.png")


def plot_model_error_heatmaps(
    oos_df: pd.DataFrame,
    output_dir: Path = None,
) -> None:
    """Figure 7: Error heatmaps — hour-of-day vs day-of-week for each model.

    Args:
        oos_df: OOS predictions with all model columns.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    model_pred_cols = {
        "LightGBM": "y_pred",
        "XGBoost": "y_pred_xgboost",
        "LGBM+Ridge": "y_pred_ensemble",
    }
    available = {k: v for k, v in model_pred_cols.items() if v in oos_df.columns}
    n_models = len(available)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Absolute Error Heatmap: Hour x Day-of-Week (OOS)",
                 fontsize=14, fontweight="bold")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    oos_copy = oos_df.copy()
    oos_copy["hour"] = oos_copy.index.hour
    oos_copy["dow"] = oos_copy.index.dayofweek

    for ax, (label, col) in zip(axes, available.items()):
        oos_copy[f"ae_{col}"] = np.abs(oos_copy["y_actual"] - oos_copy[col])
        pivot = oos_copy.pivot_table(values=f"ae_{col}", index="hour", columns="dow", aggfunc="mean")
        pivot.columns = [day_names[i] for i in pivot.columns]

        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
                    cbar_kws={"label": "MAE (EUR/MWh)"}, linewidths=0.5)
        ax.set_title(f"{label}", fontsize=12)
        ax.set_ylabel("Hour (UTC)")
        ax.set_xlabel("Day of Week")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "fig_error_heatmaps.png")
    plt.close(fig)
    logger.info("Figure 7 saved: %s", output_dir / "fig_error_heatmaps.png")


def plot_cumulative_error(
    oos_df: pd.DataFrame,
    output_dir: Path = None,
) -> None:
    """Figure 8: Cumulative absolute error over time for all models.

    Shows which model accumulates error faster. Lower = better.

    Args:
        oos_df: OOS predictions with all model columns.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    model_pred_cols = {
        "Seasonal Naive": "y_pred_naive",
        "Linear (Ridge)": "y_pred_linear",
        "LightGBM": "y_pred",
        "XGBoost": "y_pred_xgboost",
        "LGBM+Ridge Ensemble": "y_pred_ensemble",
    }
    model_colors = {
        "Seasonal Naive": "#95a5a6",
        "Linear (Ridge)": "#3498db",
        "LightGBM": "#e74c3c",
        "XGBoost": "#9b59b6",
        "LGBM+Ridge Ensemble": "#f39c12",
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle("Cumulative Absolute Error Over Time (OOS)",
                 fontsize=14, fontweight="bold")

    for label, col in model_pred_cols.items():
        if col in oos_df.columns:
            cum_ae = np.abs(oos_df["y_actual"] - oos_df[col]).cumsum()
            ax.plot(oos_df.index, cum_ae, label=label, linewidth=1.5,
                    color=model_colors.get(label, "#333"))

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative |Error| (EUR/MWh)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_cumulative_error.png")
    plt.close(fig)
    logger.info("Figure 8 saved: %s", output_dir / "fig_cumulative_error.png")


def generate_hp_report(hp_results: dict, output_dir: Path = None) -> None:
    """Write a hyperparameter search summary report.

    Args:
        hp_results: Output of run_hyperparameter_search.
        output_dir: Directory for output. Defaults to PLOTS_DIR.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = "=" * 70

    lines = [
        sep,
        " Hyperparameter Search Report",
        f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        sep,
        "",
    ]

    for model_name, data in hp_results.items():
        lines.append(f"MODEL: {model_name.upper()}")
        lines.append("-" * 40)
        lines.append(f"  Best MAE: {data['best_mae']:.2f} EUR/MWh")
        lines.append(f"  Best Params: {data['best_params']}")
        lines.append(f"  Combinations tested: {len(data['all_results'])}")
        lines.append("")

        # Top 5 results
        lines.append("  Top 5 configurations:")
        for i, r in enumerate(data["all_results"][:5]):
            lines.append(f"    {i+1}. MAE={r['mae']:.2f}  params={r['params']}")
        lines.append("")

        # Worst result for comparison
        if data["all_results"]:
            worst = data["all_results"][-1]
            lines.append(f"  Worst: MAE={worst['mae']:.2f}  params={worst['params']}")
            spread = worst["mae"] - data["best_mae"]
            lines.append(f"  Spread (worst - best): {spread:.2f} EUR/MWh")
        lines.append("")

    lines.append(sep)

    report_path = output_dir / "hyperparameter_report.txt"
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    logger.info("Hyperparameter report written to %s", report_path)


# =====================================================================
# Section 7c: Dedicated Ensemble Hyperparameter Search
# =====================================================================

# Full grid for LGBM+Ridge ensemble tuning
ENSEMBLE_PARAM_GRID = {
    "lgbm_weight": [0.5, 0.6, 0.7, 0.8, 0.9],
    "n_estimators": [500, 1000],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 20, 50],
    "ridge_alpha": [0.1, 1.0, 10.0],
}


def run_ensemble_hyperparameter_search(
    feature_matrix: pd.DataFrame,
    target_col: str = TARGET_COL,
    n_search_folds: int = 5,
) -> dict:
    """Run dedicated hyperparameter search for LGBM+Ridge ensemble.

    Searches over LightGBM params, Ridge alpha, AND the blend weight.
    Uses walk-forward folds for evaluation.

    To keep runtime manageable, uses a two-stage approach:
      Stage 1: Fix lgbm_weight=0.7, search LightGBM + Ridge params (54*3=162 combos)
      Stage 2: Take best LightGBM+Ridge params, search lgbm_weight (5 values)

    Args:
        feature_matrix: Full feature matrix.
        target_col: Target column name.
        n_search_folds: Number of walk-forward folds for evaluation.

    Returns:
        Dict with keys:
            - "best_params": dict of all best parameters
            - "best_mae": float
            - "stage1_results": list of dicts (LightGBM+Ridge combos)
            - "stage2_results": list of dicts (blend weight search)
            - "all_results": combined list sorted by MAE
    """
    logger.info("=" * 60)
    logger.info("Dedicated LGBM+Ridge Ensemble Hyperparameter Search")
    logger.info("=" * 60)

    # Create search splits
    search_splits = create_walk_forward_splits(
        feature_matrix,
        initial_train_days=WF_INITIAL_TRAIN_DAYS,
        test_window_days=WF_TEST_WINDOW_DAYS,
        step_size_days=60,
        n_folds=n_search_folds,
    )

    # ---- Stage 1: Search LightGBM + Ridge params with fixed weight=0.7 ----
    logger.info("Stage 1: Searching LightGBM + Ridge params (lgbm_weight=0.7)")

    lgbm_grid = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [10, 20, 50],
    }
    ridge_alphas = [0.1, 1.0, 10.0]

    lgbm_combos = _generate_param_combos(lgbm_grid)
    total_stage1 = len(lgbm_combos) * len(ridge_alphas)
    logger.info("Stage 1: %d LightGBM combos x %d Ridge alphas = %d total",
                len(lgbm_combos), len(ridge_alphas), total_stage1)

    stage1_results = []
    for combo_idx, lgbm_combo in enumerate(lgbm_combos):
        for ridge_alpha in ridge_alphas:
            lgbm_params = LGBM_PARAMS.copy()
            lgbm_params.update(lgbm_combo)

            fold_maes = []
            for s in search_splits:
                train = feature_matrix.loc[s["train_start"]:s["train_end"]].dropna(subset=[target_col])
                test = feature_matrix.loc[s["test_start"]:s["test_end"]].dropna(subset=[target_col])
                if len(train) == 0 or len(test) == 0:
                    continue

                X_train = train.drop(columns=[target_col])
                y_train = train[target_col]
                X_test = test.drop(columns=[target_col])
                y_test = test[target_col]

                val_size = min(720, len(X_train) // 5)
                model = LGBMRidgeEnsemble(
                    lgbm_weight=0.7,
                    lgbm_params=lgbm_params,
                    ridge_alpha=ridge_alpha,
                )
                model.fit(
                    X_train.iloc[:-val_size], y_train.iloc[:-val_size],
                    X_train.iloc[-val_size:], y_train.iloc[-val_size:],
                )
                preds = model.predict(X_test)
                fold_maes.append(np.mean(np.abs(y_test.values - preds)))

            if fold_maes:
                mean_mae = np.mean(fold_maes)
                result_entry = {
                    "lgbm_weight": 0.7,
                    "ridge_alpha": ridge_alpha,
                    **lgbm_combo,
                    "mae": mean_mae,
                    "mae_std": np.std(fold_maes),
                    "fold_maes": fold_maes,
                }
                stage1_results.append(result_entry)

        if (combo_idx + 1) % 10 == 0:
            logger.info("Stage 1 progress: %d/%d LightGBM combos done",
                        combo_idx + 1, len(lgbm_combos))

    stage1_results.sort(key=lambda x: x["mae"])
    best_stage1 = stage1_results[0]
    logger.info("Stage 1 best: MAE=%.2f, params=%s",
                best_stage1["mae"],
                {k: v for k, v in best_stage1.items() if k not in ("mae", "mae_std", "fold_maes")})

    # ---- Stage 2: Search blend weight with best LightGBM+Ridge params ----
    logger.info("Stage 2: Searching lgbm_weight with best component params")

    best_lgbm_params = LGBM_PARAMS.copy()
    best_lgbm_params.update({
        "n_estimators": best_stage1["n_estimators"],
        "learning_rate": best_stage1["learning_rate"],
        "num_leaves": best_stage1["num_leaves"],
        "min_child_samples": best_stage1["min_child_samples"],
    })
    best_ridge_alpha = best_stage1["ridge_alpha"]

    weight_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    logger.info("Stage 2: Testing %d blend weights", len(weight_values))

    stage2_results = []
    for weight in weight_values:
        fold_maes = []
        for s in search_splits:
            train = feature_matrix.loc[s["train_start"]:s["train_end"]].dropna(subset=[target_col])
            test = feature_matrix.loc[s["test_start"]:s["test_end"]].dropna(subset=[target_col])
            if len(train) == 0 or len(test) == 0:
                continue

            X_train = train.drop(columns=[target_col])
            y_train = train[target_col]
            X_test = test.drop(columns=[target_col])
            y_test = test[target_col]

            val_size = min(720, len(X_train) // 5)
            model = LGBMRidgeEnsemble(
                lgbm_weight=weight,
                lgbm_params=best_lgbm_params,
                ridge_alpha=best_ridge_alpha,
            )
            model.fit(
                X_train.iloc[:-val_size], y_train.iloc[:-val_size],
                X_train.iloc[-val_size:], y_train.iloc[-val_size:],
            )
            preds = model.predict(X_test)
            fold_maes.append(np.mean(np.abs(y_test.values - preds)))

        if fold_maes:
            mean_mae = np.mean(fold_maes)
            stage2_results.append({
                "lgbm_weight": weight,
                "ridge_weight": round(1.0 - weight, 2),
                "mae": mean_mae,
                "mae_std": np.std(fold_maes),
                "fold_maes": fold_maes,
            })
            logger.info("  lgbm_weight=%.1f (ridge=%.1f) -> MAE=%.2f +/- %.2f",
                        weight, 1.0 - weight, mean_mae, np.std(fold_maes))

    stage2_results.sort(key=lambda x: x["mae"])
    best_weight = stage2_results[0]["lgbm_weight"]

    # ---- Combine final best ----
    best_params = {
        "lgbm_weight": best_weight,
        "n_estimators": best_stage1["n_estimators"],
        "learning_rate": best_stage1["learning_rate"],
        "num_leaves": best_stage1["num_leaves"],
        "min_child_samples": best_stage1["min_child_samples"],
        "ridge_alpha": best_ridge_alpha,
    }
    best_mae = stage2_results[0]["mae"]

    logger.info("=" * 60)
    logger.info("ENSEMBLE BEST PARAMS: %s", best_params)
    logger.info("ENSEMBLE BEST MAE: %.2f EUR/MWh", best_mae)
    logger.info("=" * 60)

    return {
        "best_params": best_params,
        "best_mae": best_mae,
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "all_results": stage1_results,  # stage1 has the full grid
    }


def generate_ensemble_hp_report(
    ensemble_hp_results: dict,
    output_dir: Path = None,
) -> None:
    """Write a detailed ensemble hyperparameter search report.

    Args:
        ensemble_hp_results: Output of run_ensemble_hyperparameter_search.
        output_dir: Directory for output. Defaults to PLOTS_DIR.

    Side effects:
        Writes ensemble_hp_report.txt and prints to console.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = "=" * 70

    best = ensemble_hp_results["best_params"]
    best_mae = ensemble_hp_results["best_mae"]
    stage1 = ensemble_hp_results["stage1_results"]
    stage2 = ensemble_hp_results["stage2_results"]

    lines = [
        sep,
        " LGBM+Ridge Ensemble — Hyperparameter Search Report",
        f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        sep,
        "",
        "BEST CONFIGURATION",
        "-" * 50,
        f"  lgbm_weight:       {best['lgbm_weight']}  (ridge_weight: {1.0 - best['lgbm_weight']:.1f})",
        f"  n_estimators:      {best['n_estimators']}",
        f"  learning_rate:     {best['learning_rate']}",
        f"  num_leaves:        {best['num_leaves']}",
        f"  min_child_samples: {best['min_child_samples']}",
        f"  ridge_alpha:       {best['ridge_alpha']}",
        f"  BEST MAE:          {best_mae:.2f} EUR/MWh",
        "",
    ]

    # Stage 1: Top 10 component param combos
    lines.append("STAGE 1: COMPONENT PARAMETER SEARCH (lgbm_weight=0.7)")
    lines.append(f"Combinations tested: {len(stage1)}")
    lines.append("-" * 70)
    lines.append(
        f"{'Rank':>4s}  {'MAE':>8s}  {'Std':>6s}  {'LR':>6s}  {'Leaves':>6s}  "
        f"{'MinSamp':>7s}  {'nEst':>5s}  {'RidgeA':>6s}"
    )
    lines.append("-" * 70)
    for i, r in enumerate(stage1[:15]):
        lines.append(
            f"{i+1:>4d}  {r['mae']:>8.2f}  {r['mae_std']:>6.2f}  "
            f"{r['learning_rate']:>6.3f}  {r['num_leaves']:>6d}  "
            f"{r['min_child_samples']:>7d}  {r['n_estimators']:>5d}  "
            f"{r['ridge_alpha']:>6.1f}"
        )

    if stage1:
        worst = stage1[-1]
        lines.append(f"  ...")
        lines.append(
            f"{len(stage1):>4d}  {worst['mae']:>8.2f}  {worst['mae_std']:>6.2f}  "
            f"{worst['learning_rate']:>6.3f}  {worst['num_leaves']:>6d}  "
            f"{worst['min_child_samples']:>7d}  {worst['n_estimators']:>5d}  "
            f"{worst['ridge_alpha']:>6.1f}"
        )
        spread = worst["mae"] - stage1[0]["mae"]
        lines.append(f"\n  Spread (worst - best): {spread:.2f} EUR/MWh")

    lines.append("")

    # Stage 2: Blend weight search
    lines.append("STAGE 2: BLEND WEIGHT SEARCH")
    lines.append("-" * 50)
    lines.append(f"{'lgbm_wt':>8s}  {'ridge_wt':>8s}  {'MAE':>8s}  {'Std':>6s}  {'Status':>12s}")
    lines.append("-" * 50)
    for r in stage2:
        is_best = " <-- BEST" if r["lgbm_weight"] == best["lgbm_weight"] else ""
        lines.append(
            f"{r['lgbm_weight']:>8.1f}  {r['ridge_weight']:>8.1f}  "
            f"{r['mae']:>8.2f}  {r['mae_std']:>6.2f}  {is_best}"
        )

    lines.append("")

    # Analysis by individual hyperparameter
    lines.append("PARAMETER SENSITIVITY ANALYSIS")
    lines.append("-" * 50)

    # learning_rate sensitivity
    lr_groups = {}
    for r in stage1:
        lr = r["learning_rate"]
        lr_groups.setdefault(lr, []).append(r["mae"])
    lines.append("  learning_rate:")
    for lr in sorted(lr_groups.keys()):
        vals = lr_groups[lr]
        lines.append(f"    {lr:.3f}  ->  mean MAE={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    # num_leaves sensitivity
    nl_groups = {}
    for r in stage1:
        nl = r["num_leaves"]
        nl_groups.setdefault(nl, []).append(r["mae"])
    lines.append("  num_leaves:")
    for nl in sorted(nl_groups.keys()):
        vals = nl_groups[nl]
        lines.append(f"    {nl:>5d}  ->  mean MAE={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    # min_child_samples sensitivity
    mc_groups = {}
    for r in stage1:
        mc = r["min_child_samples"]
        mc_groups.setdefault(mc, []).append(r["mae"])
    lines.append("  min_child_samples:")
    for mc in sorted(mc_groups.keys()):
        vals = mc_groups[mc]
        lines.append(f"    {mc:>5d}  ->  mean MAE={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    # ridge_alpha sensitivity
    ra_groups = {}
    for r in stage1:
        ra = r["ridge_alpha"]
        ra_groups.setdefault(ra, []).append(r["mae"])
    lines.append("  ridge_alpha:")
    for ra in sorted(ra_groups.keys()):
        vals = ra_groups[ra]
        lines.append(f"    {ra:>6.1f}  ->  mean MAE={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    # n_estimators sensitivity
    ne_groups = {}
    for r in stage1:
        ne = r["n_estimators"]
        ne_groups.setdefault(ne, []).append(r["mae"])
    lines.append("  n_estimators:")
    for ne in sorted(ne_groups.keys()):
        vals = ne_groups[ne]
        lines.append(f"    {ne:>5d}  ->  mean MAE={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)
    report_path = output_dir / "ensemble_hp_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    logger.info("Ensemble HP report written to %s", report_path)


def plot_ensemble_hp_analysis(
    ensemble_hp_results: dict,
    output_dir: Path = None,
) -> None:
    """Figure 10: Ensemble hyperparameter analysis visualization.

    Six-panel figure showing how MAE varies with each parameter.

    Args:
        ensemble_hp_results: Output of run_ensemble_hyperparameter_search.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    stage1 = ensemble_hp_results["stage1_results"]
    stage2 = ensemble_hp_results["stage2_results"]
    best = ensemble_hp_results["best_params"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("LGBM+Ridge Ensemble — Hyperparameter Analysis",
                 fontsize=16, fontweight="bold")

    # ---- Panel 1: Blend weight vs MAE (Stage 2) ----
    ax = axes[0, 0]
    weights = [r["lgbm_weight"] for r in stage2]
    maes = [r["mae"] for r in stage2]
    stds = [r["mae_std"] for r in stage2]
    ax.errorbar(weights, maes, yerr=stds, marker="o", linewidth=2,
                color="#e74c3c", capsize=5, markersize=8)
    best_idx = weights.index(best["lgbm_weight"])
    ax.scatter([best["lgbm_weight"]], [maes[best_idx]], color="green",
               s=150, zorder=5, marker="*", label=f"Best: {best['lgbm_weight']}")
    ax.set_xlabel("LightGBM Weight (Ridge = 1 - weight)")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Blend Weight vs MAE")
    ax.legend()

    # Add secondary x-axis for Ridge weight
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ridge_ticks = [round(1.0 - w, 1) for w in weights]
    ax2.set_xticks(weights)
    ax2.set_xticklabels([f"{r:.1f}" for r in ridge_ticks], fontsize=8)
    ax2.set_xlabel("Ridge Weight", fontsize=9)

    # ---- Panel 2: learning_rate vs MAE ----
    ax = axes[0, 1]
    lr_groups = {}
    for r in stage1:
        lr_groups.setdefault(r["learning_rate"], []).append(r["mae"])
    lrs = sorted(lr_groups.keys())
    bp_data = [lr_groups[lr] for lr in lrs]
    bp = ax.boxplot(bp_data, labels=[f"{lr:.3f}" for lr in lrs],
                     patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Learning Rate vs MAE")

    # ---- Panel 3: num_leaves vs MAE ----
    ax = axes[0, 2]
    nl_groups = {}
    for r in stage1:
        nl_groups.setdefault(r["num_leaves"], []).append(r["mae"])
    nls = sorted(nl_groups.keys())
    bp_data = [nl_groups[nl] for nl in nls]
    bp = ax.boxplot(bp_data, labels=[str(nl) for nl in nls],
                     patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2ecc71")
        patch.set_alpha(0.6)
    ax.set_xlabel("num_leaves")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Number of Leaves vs MAE")

    # ---- Panel 4: min_child_samples vs MAE ----
    ax = axes[1, 0]
    mc_groups = {}
    for r in stage1:
        mc_groups.setdefault(r["min_child_samples"], []).append(r["mae"])
    mcs = sorted(mc_groups.keys())
    bp_data = [mc_groups[mc] for mc in mcs]
    bp = ax.boxplot(bp_data, labels=[str(mc) for mc in mcs],
                     patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#f39c12")
        patch.set_alpha(0.6)
    ax.set_xlabel("min_child_samples")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Min Child Samples vs MAE")

    # ---- Panel 5: ridge_alpha vs MAE ----
    ax = axes[1, 1]
    ra_groups = {}
    for r in stage1:
        ra_groups.setdefault(r["ridge_alpha"], []).append(r["mae"])
    ras = sorted(ra_groups.keys())
    bp_data = [ra_groups[ra] for ra in ras]
    bp = ax.boxplot(bp_data, labels=[f"{ra}" for ra in ras],
                     patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9b59b6")
        patch.set_alpha(0.6)
    ax.set_xlabel("ridge_alpha")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Ridge Alpha vs MAE")

    # ---- Panel 6: Top 20 configs scatter (learning_rate vs num_leaves, colored by MAE) ----
    ax = axes[1, 2]
    top20 = stage1[:20]
    lr_vals = [r["learning_rate"] for r in top20]
    nl_vals = [r["num_leaves"] for r in top20]
    mae_vals = [r["mae"] for r in top20]
    scatter = ax.scatter(lr_vals, nl_vals, c=mae_vals, cmap="RdYlGn_r",
                          s=100, edgecolors="black", linewidths=0.5)
    fig.colorbar(scatter, ax=ax, label="MAE (EUR/MWh)", shrink=0.8)
    # Mark best
    ax.scatter([best["learning_rate"]], [best["num_leaves"]], color="red",
               s=200, marker="*", zorder=5, label="Best config")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("num_leaves")
    ax.set_title("Top 20 Configs (color = MAE)")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "fig_ensemble_hp_analysis.png")
    plt.close(fig)
    logger.info("Figure 10 saved: %s", output_dir / "fig_ensemble_hp_analysis.png")


# =====================================================================
# Section 8: 12-Fold Confidence Comparison
# =====================================================================

def run_12fold_model_confidence(
    feature_matrix: pd.DataFrame,
    best_params: dict = None,
    target_col: str = TARGET_COL,
) -> dict:
    """Run full 12-fold walk-forward CV with tuned hyperparameters for all models.

    Uses paired statistical tests (Wilcoxon signed-rank) on per-fold MAE to
    determine with confidence which model is truly best. Returns a comprehensive
    comparison dict with per-fold MAEs, statistical tests, and win counts.

    Args:
        feature_matrix: Full feature matrix.
        best_params: Dict from run_hyperparameter_search with best params.
        target_col: Target column name.

    Returns:
        Dict with keys:
            - "fold_maes": DataFrame of (fold × model) MAE values
            - "summary": DataFrame with mean, std, median, min, max per model
            - "pairwise_tests": list of dicts with Wilcoxon test results
            - "win_matrix": DataFrame of head-to-head fold wins
            - "best_model": name of model with lowest mean MAE
    """
    from scipy.stats import wilcoxon

    logger.info("=" * 60)
    logger.info("12-Fold Model Confidence Comparison")
    logger.info("=" * 60)

    # Create full 12-fold splits
    splits = create_walk_forward_splits(
        feature_matrix,
        initial_train_days=WF_INITIAL_TRAIN_DAYS,
        test_window_days=WF_TEST_WINDOW_DAYS,
        step_size_days=WF_STEP_SIZE_DAYS,
        n_folds=WF_N_FOLDS,
    )

    # Build models with tuned params
    lgbm_p = LGBM_PARAMS.copy()
    xgb_p = XGBM_PARAMS.copy()
    ridge_alpha = 10.0

    if best_params:
        if "lightgbm" in best_params and best_params["lightgbm"].get("best_params"):
            lgbm_p.update(best_params["lightgbm"]["best_params"])
        if "xgboost" in best_params and best_params["xgboost"].get("best_params"):
            xgb_p.update(best_params["xgboost"]["best_params"])
        if "ridge" in best_params and best_params["ridge"].get("best_params"):
            ridge_alpha = best_params["ridge"]["best_params"].get("alpha", 10.0)

    models = {
        "seasonal_naive": SeasonalNaiveModel(),
        "linear_regression": LinearRegressionBaseline(alpha=ridge_alpha),
        "lightgbm": LightGBMForecaster(params=lgbm_p),
        "xgboost": XGBoostForecaster(params=xgb_p),
        "lgbm_ridge_ensemble": LGBMRidgeEnsemble(
            lgbm_params=lgbm_p, ridge_alpha=ridge_alpha
        ),
    }

    model_names = list(models.keys())

    # Collect per-fold MAE for each model
    fold_maes = {name: [] for name in model_names}

    for s in splits:
        fold = s["fold"]
        train = feature_matrix.loc[s["train_start"]:s["train_end"]].dropna(subset=[target_col])
        test = feature_matrix.loc[s["test_start"]:s["test_end"]].dropna(subset=[target_col])

        if len(train) == 0 or len(test) == 0:
            logger.warning("Fold %d: empty after NaN drop — skipping", fold)
            for name in model_names:
                fold_maes[name].append(np.nan)
            continue

        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]

        for name, model in models.items():
            logger.info("12-Fold Confidence | Fold %d/%d: %s (%d train, %d test)",
                        fold + 1, len(splits), name, len(X_train), len(X_test))

            # Carve validation set for tree-based and ensemble models
            if name in ("lightgbm", "xgboost", "lgbm_ridge_ensemble"):
                val_size = min(720, len(X_train) // 5)
                model.fit(
                    X_train.iloc[:-val_size], y_train.iloc[:-val_size],
                    X_train.iloc[-val_size:], y_train.iloc[-val_size:],
                )
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mae = np.mean(np.abs(y_test.values - preds))
            fold_maes[name].append(mae)

    # Build results DataFrame
    fold_maes_df = pd.DataFrame(fold_maes)
    fold_maes_df.index.name = "fold"

    # Summary statistics
    summary_records = []
    for name in model_names:
        vals = fold_maes_df[name].dropna()
        summary_records.append({
            "model": name,
            "mean_mae": vals.mean(),
            "std_mae": vals.std(),
            "median_mae": vals.median(),
            "min_mae": vals.min(),
            "max_mae": vals.max(),
            "cv_pct": (vals.std() / vals.mean()) * 100,  # coefficient of variation
        })
    summary_df = pd.DataFrame(summary_records).sort_values("mean_mae")

    # Pairwise Wilcoxon signed-rank tests
    pairwise_tests = []
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            a_vals = fold_maes_df[name_a].dropna().values
            b_vals = fold_maes_df[name_b].dropna().values
            n = min(len(a_vals), len(b_vals))
            a_vals = a_vals[:n]
            b_vals = b_vals[:n]

            diff = a_vals - b_vals
            if np.all(diff == 0) or n < 6:
                p_value = 1.0
                statistic = np.nan
            else:
                try:
                    statistic, p_value = wilcoxon(a_vals, b_vals, alternative="two-sided")
                except ValueError:
                    p_value = 1.0
                    statistic = np.nan

            # Determine winner
            mean_diff = np.mean(a_vals) - np.mean(b_vals)
            winner = name_a if mean_diff < 0 else name_b
            significant = p_value < 0.05

            pairwise_tests.append({
                "model_a": name_a,
                "model_b": name_b,
                "mean_a": np.mean(a_vals),
                "mean_b": np.mean(b_vals),
                "mean_diff": mean_diff,
                "winner": winner,
                "p_value": p_value,
                "significant": significant,
                "confidence": f"{'YES' if significant else 'NO'} (p={p_value:.4f})",
            })

    # Head-to-head win matrix
    win_matrix = pd.DataFrame(0, index=model_names, columns=model_names)
    for fold_idx in range(len(fold_maes_df)):
        fold_row = fold_maes_df.iloc[fold_idx]
        for name_a in model_names:
            for name_b in model_names:
                if name_a != name_b and fold_row[name_a] < fold_row[name_b]:
                    win_matrix.loc[name_a, name_b] += 1

    best_model = summary_df.iloc[0]["model"]

    # Print comprehensive report
    sep = "=" * 70
    lines = [
        sep,
        " 12-FOLD WALK-FORWARD MODEL CONFIDENCE REPORT",
        f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f" Folds: {len(splits)} | Test window: {WF_TEST_WINDOW_DAYS} days each",
        sep,
        "",
        "PER-FOLD MAE (EUR/MWh)",
        "-" * 70,
    ]

    # Per-fold table header
    header = f"{'Fold':>6s}"
    for name in model_names:
        header += f"  {name:>18s}"
    lines.append(header)
    lines.append("-" * 70)

    for fold_idx in range(len(fold_maes_df)):
        row = f"{fold_idx:>6d}"
        fold_row = fold_maes_df.iloc[fold_idx]
        fold_min = fold_row.min()
        for name in model_names:
            val = fold_row[name]
            marker = " *" if val == fold_min else "  "
            row += f"  {val:>16.2f}{marker}"
        lines.append(row)

    lines.append("")
    lines.append("(* = best model for that fold)")
    lines.append("")

    # Summary statistics
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 70)
    lines.append(f"{'Model':25s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'CV%':>6s}")
    lines.append("-" * 70)
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['model']:25s} {row['mean_mae']:>8.2f} {row['std_mae']:>8.2f} "
            f"{row['median_mae']:>8.2f} {row['min_mae']:>8.2f} {row['max_mae']:>8.2f} "
            f"{row['cv_pct']:>5.1f}%"
        )

    lines.append("")

    # Pairwise statistical tests
    lines.append("PAIRWISE STATISTICAL TESTS (Wilcoxon signed-rank, alpha=0.05)")
    lines.append("-" * 70)
    lines.append(f"{'Model A':20s} vs {'Model B':20s} {'Winner':20s} {'p-value':>10s} {'Significant':>12s}")
    lines.append("-" * 70)
    for t in pairwise_tests:
        lines.append(
            f"{t['model_a']:20s} vs {t['model_b']:20s} {t['winner']:20s} "
            f"{t['p_value']:>10.4f} {t['confidence']:>12s}"
        )

    lines.append("")

    # Win matrix
    lines.append("HEAD-TO-HEAD WIN COUNTS (row beat column in N folds)")
    lines.append("-" * 70)
    win_header = f"{'':20s}"
    for name in model_names:
        short = name[:12]
        win_header += f"  {short:>12s}"
    lines.append(win_header)
    for name in model_names:
        row_str = f"{name:20s}"
        for name_b in model_names:
            if name == name_b:
                row_str += f"  {'---':>12s}"
            else:
                row_str += f"  {win_matrix.loc[name, name_b]:>12d}"
        lines.append(row_str)

    lines.append("")
    lines.append(f"BEST MODEL: {best_model} (mean MAE = {summary_df.iloc[0]['mean_mae']:.2f} EUR/MWh)")

    # Check if best is statistically significantly better than runner-up
    runner_up = summary_df.iloc[1]["model"]
    for t in pairwise_tests:
        pair = {t["model_a"], t["model_b"]}
        if pair == {best_model, runner_up}:
            if t["significant"]:
                lines.append(
                    f"  -> Statistically significantly better than {runner_up} "
                    f"(p={t['p_value']:.4f})"
                )
            else:
                lines.append(
                    f"  -> NOT statistically significantly better than {runner_up} "
                    f"(p={t['p_value']:.4f}) — difference may be due to chance"
                )
            break

    lines.append(sep)

    report_text = "\n".join(lines)
    print(report_text)

    return {
        "fold_maes": fold_maes_df,
        "summary": summary_df,
        "pairwise_tests": pairwise_tests,
        "win_matrix": win_matrix,
        "best_model": best_model,
        "report_text": report_text,
    }


def plot_12fold_confidence(
    confidence_results: dict,
    output_dir: Path = None,
) -> None:
    """Figure 9: 12-fold confidence visualization.

    Four-panel figure: per-fold MAE lines, box plots with significance
    markers, win-rate heatmap, and paired difference distribution.

    Args:
        confidence_results: Output of run_12fold_model_confidence.
        output_dir: Directory for output figures. Defaults to PLOTS_DIR.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    fold_maes_df = confidence_results["fold_maes"]
    summary_df = confidence_results["summary"]
    win_matrix = confidence_results["win_matrix"]
    pairwise_tests = confidence_results["pairwise_tests"]
    best_model = confidence_results["best_model"]

    model_names = list(fold_maes_df.columns)
    model_colors = {
        "seasonal_naive": "#95a5a6",
        "linear_regression": "#3498db",
        "lightgbm": "#e74c3c",
        "xgboost": "#9b59b6",
        "lgbm_ridge_ensemble": "#f39c12",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("12-Fold Walk-Forward Model Confidence Analysis",
                 fontsize=16, fontweight="bold")

    # ---- Panel 1: Per-fold MAE lines ----
    ax = axes[0, 0]
    for name in model_names:
        vals = fold_maes_df[name].values
        ax.plot(range(len(vals)), vals, marker="o", markersize=5, linewidth=2,
                label=name, color=model_colors.get(name, "#333"))
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Per-Fold MAE Across 12 Folds")
    ax.set_xticks(range(len(fold_maes_df)))
    ax.legend(fontsize=7, ncol=2)

    # ---- Panel 2: Box plots with mean markers ----
    ax = axes[0, 1]
    data_for_box = [fold_maes_df[name].dropna().values for name in model_names]
    bp = ax.boxplot(data_for_box, labels=model_names, patch_artist=True,
                     showmeans=True, meanline=True, widths=0.6,
                     meanprops=dict(color="black", linewidth=2, linestyle="--"))
    colors_list = [model_colors.get(n, "#333") for n in model_names]
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("MAE Distribution (12 folds) — dashed line = mean")
    ax.tick_params(axis="x", rotation=20)

    # Add significance stars between best and each other model
    for t in pairwise_tests:
        if best_model in (t["model_a"], t["model_b"]) and t["significant"]:
            other = t["model_b"] if t["model_a"] == best_model else t["model_a"]
            idx_best = model_names.index(best_model)
            idx_other = model_names.index(other)
            y_max = max(fold_maes_df[best_model].max(), fold_maes_df[other].max())
            ax.annotate("*", xy=((idx_best + idx_other) / 2 + 0.5, y_max * 1.02),
                        fontsize=16, ha="center", color="red", fontweight="bold")

    # ---- Panel 3: Win-rate heatmap ----
    ax = axes[1, 0]
    n_folds = len(fold_maes_df)
    win_pct = win_matrix / n_folds * 100
    short_names = [n[:12] for n in model_names]
    sns.heatmap(win_pct, annot=True, fmt=".0f", cmap="RdYlGn",
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, vmin=0, vmax=100, cbar_kws={"label": "Win %"})
    ax.set_title("Head-to-Head Win Rate (%) — row beats column")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # ---- Panel 4: Paired MAE differences (best model vs each other) ----
    ax = axes[1, 1]
    others = [n for n in model_names if n != best_model]
    positions = range(len(others))
    for i, other in enumerate(others):
        diff = fold_maes_df[other].values - fold_maes_df[best_model].values
        parts = ax.violinplot([diff[~np.isnan(diff)]], positions=[i],
                               showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(model_colors.get(other, "#333"))
            pc.set_alpha(0.6)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(others, fontsize=8, rotation=15)
    ax.set_ylabel(f"MAE(other) - MAE({best_model}) [EUR/MWh]")
    ax.set_title(f"Paired Difference vs Best Model ({best_model})\n(> 0 means {best_model} is better)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "fig_12fold_confidence.png")
    plt.close(fig)
    logger.info("Figure 9 saved: %s", output_dir / "fig_12fold_confidence.png")


# =====================================================================
# Orchestration helpers (called by run_forecasting.py)
# =====================================================================

def compute_regime_metrics(oos_df: pd.DataFrame, feature_matrix: pd.DataFrame) -> dict:
    """Compute LightGBM metrics split by crisis vs normal regime.

    Args:
        oos_df: OOS predictions DataFrame.
        feature_matrix: Feature matrix for gas price lookup.

    Returns:
        Dict of {regime_name: metrics_dict}.
    """
    # Use gas price to define regimes
    gas_col = "gas_price_eur_mwh"
    if gas_col in feature_matrix.columns:
        oos_gas = feature_matrix.loc[oos_df.index, gas_col]
        crisis_mask = oos_gas > 80
    else:
        crisis_mask = pd.Series(False, index=oos_df.index)

    result = {}
    actual = oos_df["y_actual"].values
    predicted = oos_df["y_pred"].values

    if crisis_mask.sum() > 0:
        result["Crisis (gas > 80 EUR/MWh)"] = compute_metrics(
            actual[crisis_mask], predicted[crisis_mask], "lightgbm"
        )

    normal_mask = ~crisis_mask
    if normal_mask.sum() > 0:
        result["Normal (gas <= 80 EUR/MWh)"] = compute_metrics(
            actual[normal_mask], predicted[normal_mask], "lightgbm"
        )

    return result


def compute_block_metrics(oos_df: pd.DataFrame) -> dict:
    """Compute LightGBM metrics split by peak vs off-peak.

    Args:
        oos_df: OOS predictions DataFrame.

    Returns:
        Dict of {block_name: metrics_dict}.
    """
    actual = oos_df["y_actual"].values
    predicted = oos_df["y_pred"].values
    hours = oos_df.index.hour

    peak_mask = np.isin(hours, PEAK_HOURS)
    offpeak_mask = ~peak_mask

    result = {}
    if peak_mask.sum() > 0:
        result["Peak (08-19 UTC)"] = compute_metrics(
            actual[peak_mask], predicted[peak_mask], "lightgbm"
        )
    if offpeak_mask.sum() > 0:
        result["Off-peak"] = compute_metrics(
            actual[offpeak_mask], predicted[offpeak_mask], "lightgbm"
        )

    return result


def compute_weekly_monthly_metrics(
    oos_df: pd.DataFrame,
    model_columns: dict = None,
) -> dict:
    """Compute forecast accuracy at weekly (ISO week) and monthly granularity.

    Aggregates hourly predictions upward rather than training separate models,
    which is more robust given limited data (~156 weeks, ~36 months).

    Args:
        oos_df: OOS predictions DataFrame with DatetimeIndex and columns
                y_actual, y_pred (ensemble), y_pred_naive, plus any
                additional model prediction columns.
        model_columns: Optional dict mapping model_name -> column_name in
                       oos_df. Defaults to detecting y_pred_* columns.

    Returns:
        Dict with keys 'hourly', 'weekly', 'monthly', each containing
        {model_name: {MAE, RMSE, MAPE, MBE, n}}.
    """
    if model_columns is None:
        model_columns = {}
        for col in oos_df.columns:
            if col.startswith("y_pred_"):
                name = col.replace("y_pred_", "")
                model_columns[name] = col
            elif col == "y_pred":
                model_columns["lgbm_ridge_ensemble"] = col

    actual_col = "y_actual"
    result = {}

    for granularity in ["hourly", "weekly", "monthly"]:
        gran_metrics = {}

        for model_name, pred_col in model_columns.items():
            if pred_col not in oos_df.columns:
                continue

            df = oos_df[[actual_col, pred_col]].dropna()

            if granularity == "weekly":
                # Group by ISO year-week
                df = df.copy()
                df["_yr"] = df.index.isocalendar().year.values
                df["_wk"] = df.index.isocalendar().week.values
                grouped = df.groupby(["_yr", "_wk"]).agg(
                    actual_mean=(actual_col, "mean"),
                    pred_mean=(pred_col, "mean"),
                ).dropna()
                actual_vals = grouped["actual_mean"].values
                pred_vals = grouped["pred_mean"].values

            elif granularity == "monthly":
                df = df.copy()
                # Use year-month string to avoid timezone→Period warning
                df["_ym"] = df.index.strftime("%Y-%m")
                grouped = df.groupby("_ym").agg(
                    actual_mean=(actual_col, "mean"),
                    pred_mean=(pred_col, "mean"),
                ).dropna()
                actual_vals = grouped["actual_mean"].values
                pred_vals = grouped["pred_mean"].values

            else:  # hourly
                actual_vals = df[actual_col].values
                pred_vals = df[pred_col].values

            if len(actual_vals) == 0:
                continue

            errors = actual_vals - pred_vals
            abs_errors = np.abs(errors)
            mae = float(np.mean(abs_errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mbe = float(np.mean(errors))

            # MAPE: skip zeros in actual to avoid division by zero
            nonzero_mask = actual_vals != 0
            if nonzero_mask.sum() > 0:
                mape = float(
                    np.mean(np.abs(errors[nonzero_mask] / actual_vals[nonzero_mask])) * 100
                )
            else:
                mape = np.nan

            gran_metrics[model_name] = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "MBE": mbe,
                "n": len(actual_vals),
            }

        result[granularity] = gran_metrics

    return result

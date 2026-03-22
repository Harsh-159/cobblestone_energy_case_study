"""
AutoGluon-based forecaster for DE_LU day-ahead electricity price forecasting.

Wraps autogluon.tabular.TabularPredictor with the same interface as the
existing model classes (fit/predict/save/load) for seamless integration
into the walk-forward CV and OOS evaluation pipeline.

AutoGluon automatically searches over:
    - Model types (LightGBM, XGBoost, CatBoost, Neural Nets, Linear, KNN, etc.)
    - Hyperparameters for each model
    - Ensemble stacking and bagging

Inputs:
    Feature matrix from features.py (same as all other models)

Outputs:
    Predictions, leaderboard, and best model summary
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion import PROJECT_ROOT, OUTPUTS_DIR, LOGS_DIR
from src.models import TARGET_COL, MODELS_DIR, PLOTS_DIR

logger = logging.getLogger("autogluon_forecaster")

# AutoGluon save directory
AUTOGLUON_DIR = MODELS_DIR / "autogluon"


def _setup_logging() -> None:
    """Configure logging for AutoGluon forecaster."""
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

    fh = logging.FileHandler(LOGS_DIR / "autogluon.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


class AutoGluonForecaster:
    """AutoGluon TabularPredictor wrapper for DA price forecasting.

    Automatically searches over model types (LightGBM, XGBoost, CatBoost,
    Neural Nets, Linear, etc.) and their hyperparameters. Supports ensemble
    stacking.

    Follows the same interface as LightGBMForecaster, XGBoostForecaster, etc.
    """

    def __init__(self, time_limit: int = 300, presets: str = "best_quality",
                 label: str = TARGET_COL, save_path: str = None):
        """Initialize AutoGluon forecaster.

        Args:
            time_limit: Maximum training time in seconds.
            presets: AutoGluon preset ('best_quality', 'high_quality',
                     'medium_quality', 'good_quality').
            label: Target column name.
            save_path: Directory to save AutoGluon artifacts. Defaults to
                       models/autogluon.
        """
        self.name = "autogluon"
        self.time_limit = time_limit
        self.presets = presets
        self.label = label
        self.save_path = save_path or str(AUTOGLUON_DIR)
        self.predictor = None
        self.feature_names = None
        self._leaderboard = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Fit AutoGluon on training data.

        Args:
            X_train: Training feature matrix (target column excluded).
            y_train: Training target values.
            X_val: Optional validation features for tuning.
            y_val: Optional validation target.

        Side effects:
            Sets self.predictor, self.feature_names, self._leaderboard.
        """
        from autogluon.tabular import TabularPredictor

        self.feature_names = [c for c in X_train.columns if c != self.label]

        # AutoGluon expects a single DataFrame with the label column
        train_df = X_train[self.feature_names].copy()
        train_df[self.label] = y_train.values

        # Clean inf values (AutoGluon handles NaN but not inf)
        train_df = train_df.replace([np.inf, -np.inf], np.nan)

        # Prepare validation data if provided
        tuning_data = None
        if X_val is not None and y_val is not None:
            val_df = X_val[self.feature_names].copy()
            val_df[self.label] = y_val.values
            val_df = val_df.replace([np.inf, -np.inf], np.nan)
            tuning_data = val_df

        # Clean up previous AutoGluon save directory (force full removal)
        save_path = Path(self.save_path)
        if save_path.exists():
            shutil.rmtree(save_path, ignore_errors=True)
        # Double-check it's gone; recreate empty dir
        if save_path.exists():
            import os
            os.system(f'rm -rf "{save_path}"')
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Fitting AutoGluon: time_limit=%ds, presets=%s, "
            "train=%d rows, features=%d",
            self.time_limit, self.presets, len(train_df), len(self.feature_names),
        )

        # best_quality uses bagging — combine train+val into one dataset
        # and let AutoGluon handle internal CV splits
        if tuning_data is not None:
            import pandas as _pd
            full_train_df = _pd.concat([train_df, tuning_data], ignore_index=True)
        else:
            full_train_df = train_df

        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric="mean_absolute_error",
            path=str(save_path),
            verbosity=2,
        ).fit(
            train_data=full_train_df,
            time_limit=self.time_limit,
            presets=self.presets,
            hyperparameters={'GBM': {}, 'CAT': {}, 'XGB': {}, 'RF': {}, 'XT': {}}, # explicitly list only trees
            dynamic_stacking=False,
        )

        self._leaderboard = self.predictor.leaderboard(silent=True)
        logger.info("AutoGluon fit complete. Models trained: %d", len(self._leaderboard))
        logger.info("Best model: %s", self.predictor.model_best)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using the best AutoGluon model.

        Args:
            X_test: Test feature matrix.

        Returns:
            Numpy array of predictions.
        """
        test_df = X_test[self.feature_names].copy()
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        return self.predictor.predict(test_df).values

    def leaderboard(self) -> pd.DataFrame:
        """Return the AutoGluon model leaderboard.

        Returns:
            DataFrame with model names, scores, and training times.
        """
        if self._leaderboard is not None:
            return self._leaderboard
        if self.predictor is not None:
            return self.predictor.leaderboard(silent=True)
        return pd.DataFrame()

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Return feature importance from AutoGluon's best model.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with columns: feature, importance, importance_pct.
        """
        if self.predictor is None:
            return pd.DataFrame()

        try:
            imp = self.predictor.feature_importance(
                data=None, silent=True
            )
            imp_df = pd.DataFrame({
                "feature": imp.index,
                "importance": imp["importance"].values,
            })
            imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
            total = imp_df["importance"].sum()
            imp_df["importance_pct"] = (
                imp_df["importance"] / total * 100 if total > 0 else 0
            )
            return imp_df.reset_index(drop=True)
        except Exception as e:
            logger.warning("Could not get feature importance: %s", e)
            return pd.DataFrame()

    def save(self, path: Path = None) -> None:
        """Save AutoGluon predictor. Already saved during fit, but this
        logs the location.

        Args:
            path: Ignored (AutoGluon saves to self.save_path during fit).
        """
        logger.info("AutoGluon model saved at %s", self.save_path)

    @classmethod
    def load(cls, path: str = None) -> "AutoGluonForecaster":
        """Load a saved AutoGluon predictor.

        Args:
            path: Directory where AutoGluon was saved.

        Returns:
            AutoGluonForecaster instance with loaded predictor.
        """
        from autogluon.tabular import TabularPredictor

        path = path or str(AUTOGLUON_DIR)
        instance = cls()
        instance.predictor = TabularPredictor.load(path)
        instance.feature_names = [
            c for c in instance.predictor.feature_metadata_in.get_features()
            if c != TARGET_COL
        ]
        instance._leaderboard = instance.predictor.leaderboard(silent=True)
        return instance


def generate_autogluon_report(
    ag_model: AutoGluonForecaster,
    oos_metrics: dict,
    weekly_monthly_metrics: dict = None,
    output_dir: Path = None,
) -> str:
    """Generate a comprehensive AutoGluon training report.

    Args:
        ag_model: Fitted AutoGluonForecaster instance.
        oos_metrics: Dict of OOS metrics for all models including AutoGluon.
        weekly_monthly_metrics: Dict with weekly/monthly accuracy metrics.
        output_dir: Output directory. Defaults to PLOTS_DIR.

    Returns:
        Report text string.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    lines = [
        sep,
        " AutoGluon Training & Comprehensive Performance Report",
        f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        sep,
        "",
    ]

    # Section 1: AutoGluon model leaderboard
    lines.append("AUTOGLUON MODEL LEADERBOARD")
    lines.append("-" * 70)

    lb = ag_model.leaderboard()
    if len(lb) > 0:
        # Select key columns
        display_cols = []
        for col in ["model", "score_val", "pred_time_val", "fit_time", "can_infer"]:
            if col in lb.columns:
                display_cols.append(col)

        lines.append(
            f"{'Rank':>4s}  {'Model':35s}  {'Val Score':>12s}  "
            f"{'Fit Time(s)':>12s}  {'Pred Time(s)':>12s}"
        )
        lines.append("-" * 70)

        for i, (_, row) in enumerate(lb.iterrows()):
            model_name = str(row.get("model", "?"))[:35]
            score = row.get("score_val", 0)
            fit_t = row.get("fit_time", 0)
            pred_t = row.get("pred_time_val", 0)
            marker = " <-- BEST" if i == 0 else ""
            lines.append(
                f"{i+1:>4d}  {model_name:35s}  {score:>12.4f}  "
                f"{fit_t:>12.1f}  {pred_t:>12.4f}{marker}"
            )

        lines.append(f"\nTotal models evaluated: {len(lb)}")
        lines.append(f"Best model: {ag_model.predictor.model_best}")
    else:
        lines.append("  No leaderboard available.")

    lines.append("")

    # Section 2: OOS performance comparison
    lines.append("OOS PERFORMANCE COMPARISON (Hourly)")
    lines.append("-" * 70)
    lines.append(
        f"{'Model':30s}  {'MAE':>8s}  {'RMSE':>8s}  {'P95_AE':>8s}  "
        f"{'Dir.Acc':>8s}  {'Skill':>8s}"
    )
    lines.append("-" * 70)

    for model_name, m in sorted(oos_metrics.items(), key=lambda x: x[1].get("MAE", 999)):
        mae = f"{m.get('MAE', 0):.1f}"
        rmse = f"{m.get('RMSE', 0):.1f}"
        p95 = f"{m.get('p95_AE', 0):.1f}"
        da = f"{m.get('directional_accuracy', 0):.1f}%" if pd.notna(m.get("directional_accuracy")) else "N/A"
        skill = f"{m.get('skill_score_vs_naive', 0):+.2f}" if pd.notna(m.get("skill_score_vs_naive")) else "N/A"
        lines.append(
            f"{model_name:30s}  {mae:>8s}  {rmse:>8s}  {p95:>8s}  "
            f"{da:>8s}  {skill:>8s}"
        )

    lines.append("")

    # Section 3: Weekly / Monthly accuracy
    if weekly_monthly_metrics:
        lines.append("MULTI-GRANULARITY FORECAST ACCURACY")
        lines.append("-" * 70)

        for granularity in ["hourly", "weekly", "monthly"]:
            if granularity in weekly_monthly_metrics:
                gran_data = weekly_monthly_metrics[granularity]
                lines.append(f"\n  {granularity.upper()} ACCURACY:")
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
        lines.append("  Note: Weekly = ISO week averages, Monthly = calendar month averages")
        lines.append("        Aggregated from hourly forecasts (not separate models)")
        lines.append("        OOS period Jun-Dec 2024 gives ~30 weeks and 7 months")

    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)
    report_path = output_dir / "comprehensive_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    logger.info("Comprehensive report written to %s", report_path)
    return report_text

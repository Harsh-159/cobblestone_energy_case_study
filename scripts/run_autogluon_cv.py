#!/usr/bin/env python3
"""
Run AutoGluon through the same 12-fold walk-forward CV as the ensemble.
Estimated runtime: ~60 minutes (12 folds × 300s each).
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.features import build_feature_matrix
from src.ingestion import DATA_PROCESSED, OUTPUTS_DIR
from src.models import (
    TARGET_COL, PLOTS_DIR,
    create_walk_forward_splits,
    compute_metrics,
    _setup_logging,
)
from src.autogluon_forecaster import AutoGluonForecaster

logger = logging.getLogger("autogluon_cv")


def main():
    _setup_logging()
    sep = "=" * 60

    logger.info(sep)
    logger.info("AutoGluon Walk-Forward CV (12 folds × 300s each)")
    logger.info(sep)

    # Load feature matrix
    parquet_path = DATA_PROCESSED / "de_power_dataset.parquet"
    df = pd.read_parquet(parquet_path)
    feature_matrix = build_feature_matrix(df)
    logger.info("Feature matrix: %d rows x %d columns",
                feature_matrix.shape[0], feature_matrix.shape[1])

    # Create same walk-forward splits as the main pipeline
    splits = create_walk_forward_splits(feature_matrix)
    logger.info("Walk-forward splits: %d folds", len(splits))

    fold_results = []

    for fold_idx, s in enumerate(splits):
        logger.info(sep)
        logger.info("FOLD %d/%d: train %s to %s, test %s to %s",
                     fold_idx + 1, len(splits),
                     s["train_start"].strftime("%Y-%m-%d"),
                     s["train_end"].strftime("%Y-%m-%d"),
                     s["test_start"].strftime("%Y-%m-%d"),
                     s["test_end"].strftime("%Y-%m-%d"))

        train = feature_matrix.loc[s["train_start"]:s["train_end"]].dropna(subset=[TARGET_COL])
        test = feature_matrix.loc[s["test_start"]:s["test_end"]].dropna(subset=[TARGET_COL])

        if len(train) == 0 or len(test) == 0:
            logger.warning("Fold %d: empty train or test, skipping", fold_idx + 1)
            continue

        X_train = train.drop(columns=[TARGET_COL])
        y_train = train[TARGET_COL]
        X_test = test.drop(columns=[TARGET_COL])
        y_test = test[TARGET_COL]

        # Use a unique save path per fold to avoid conflicts
        fold_save_path = str(PROJECT_ROOT / "models" / f"autogluon_cv_fold{fold_idx}")

        ag = AutoGluonForecaster(
            time_limit=300,
            presets="best_quality",
            save_path=fold_save_path,
        )

        try:
            ag.fit(X_train, y_train)
            preds = ag.predict(X_test)

            fold_mae = np.mean(np.abs(y_test.values - preds))
            fold_rmse = np.sqrt(np.mean((y_test.values - preds) ** 2))

            # Directional accuracy
            if "price_lag_168h" in X_test.columns:
                lag = X_test["price_lag_168h"].values
                actual_dir = (y_test.values > lag).astype(int)
                pred_dir = (preds > lag).astype(int)
                dir_acc = np.mean(actual_dir == pred_dir) * 100
            else:
                dir_acc = np.nan

            fold_results.append({
                "fold": fold_idx + 1,
                "mae": fold_mae,
                "rmse": fold_rmse,
                "dir_acc": dir_acc,
                "best_model": ag.predictor.model_best,
                "n_models": len(ag.leaderboard()),
                "n_test": len(test),
            })

            logger.info("  Fold %d: MAE=%.2f, RMSE=%.2f, DirAcc=%.1f%%, best=%s (%d models)",
                         fold_idx + 1, fold_mae, fold_rmse, dir_acc,
                         ag.predictor.model_best, len(ag.leaderboard()))

        except Exception as e:
            logger.error("  Fold %d FAILED: %s", fold_idx + 1, e)
            fold_results.append({
                "fold": fold_idx + 1,
                "mae": np.nan,
                "rmse": np.nan,
                "dir_acc": np.nan,
                "best_model": "FAILED",
                "n_models": 0,
                "n_test": len(test),
            })

        # Clean up fold model directory to save disk space
        import shutil
        fold_path = Path(fold_save_path)
        if fold_path.exists():
            shutil.rmtree(fold_path, ignore_errors=True)

    # ---- Summary ----
    logger.info(sep)
    logger.info("AUTOGLUON 12-FOLD WALK-FORWARD CV RESULTS")
    logger.info(sep)

    results_df = pd.DataFrame(fold_results)
    valid = results_df.dropna(subset=["mae"])

    logger.info("")
    logger.info("%-6s  %8s  %8s  %8s  %s",
                "Fold", "MAE", "RMSE", "DirAcc", "Best Model")
    logger.info("-" * 70)
    for _, row in results_df.iterrows():
        logger.info("%-6d  %8.2f  %8.2f  %7.1f%%  %s",
                     row["fold"], row["mae"], row["rmse"],
                     row["dir_acc"], row["best_model"])

    mean_mae = valid["mae"].mean()
    std_mae = valid["mae"].std()
    mean_rmse = valid["rmse"].mean()
    mean_dir = valid["dir_acc"].mean()

    logger.info("-" * 70)
    logger.info("MEAN    %8.2f  %8.2f  %7.1f%%", mean_mae, mean_rmse, mean_dir)
    logger.info("STD     %8.2f", std_mae)
    logger.info("")
    logger.info("AutoGluon CV MAE: %.2f +/- %.2f EUR/MWh", mean_mae, std_mae)
    logger.info("AutoGluon CV DirAcc: %.1f%%", mean_dir)
    logger.info("")
    logger.info("For comparison (from previous run):")
    logger.info("  LGBM+Ridge Ensemble CV MAE: 42.6 +/- 19.3")
    logger.info("  Seasonal Naive CV MAE:      86.2 +/- 31.9")
    logger.info(sep)

    # Save results
    results_path = OUTPUTS_DIR / "autogluon_cv_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Results saved to %s", results_path)

    return results_df


if __name__ == "__main__":
    main()

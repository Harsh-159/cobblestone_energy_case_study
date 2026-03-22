"""
Wrapper for AutoGluon TimeSeries predicting European Power Prices.
"""
import logging
from pathlib import Path
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

logger = logging.getLogger("forecasting")


class AutoGluonTimeSeriesForecaster:
    def __init__(self, target_col="da_price_eur_mwh", known_covariates=None, prediction_length=24, path="models/autogluon_ts"):
        self.target_col = target_col
        self.known_covariates = known_covariates or []
        self.prediction_length = prediction_length
        self.path = path
        self.predictor = None

    def _prepare_ts_data(self, df: pd.DataFrame) -> TimeSeriesDataFrame:
        """Convert a standard pandas DataFrame to a TimeSeriesDataFrame.
        AutoGluon TimeSeries requires an 'item_id' column for multiple series; 
        we only have one DE_LU price series.
        """
        temp_df = df.copy()
        
        # AutoGluon TS expects tz-naive datetime64. ENTSO-E data is tz-aware UTC.
        if temp_df.index.tz is not None:
            temp_df.index = temp_df.index.tz_localize(None)
            
        # Guarantee hourly frequency before AutoGluon conversion
        # Drop duplicates and resample to fill any missing hours
        temp_df = temp_df[~temp_df.index.duplicated(keep="first")]
        temp_df = temp_df.resample("h").ffill()
        
        # Add the mandated item_id column for a single time series
        temp_df["item_id"] = "DE_LU"
        
        # Reset index to bring timestamp into a column
        temp_df = temp_df.reset_index()
        
        # Construct the TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            temp_df,
            id_column="item_id",
            timestamp_column="timestamp_utc"
        )
        return ts_df

    def fit(self, train_df: pd.DataFrame, time_limit: int = 600, presets: str = "medium_quality"):
        """Train the TimeSeriesPredictor.
        
        Args:
            train_df: Full training dataframe (features + target) index by timestamp.
            time_limit: Time limit for AutoGluon in seconds.
            presets: Quality preset (e.g. fast_training, medium_quality, best_quality).
        """
        logger.info("Preparing data for AutoGluon TimeSeries...")
        train_data = self._prepare_ts_data(train_df)
        
        logger.info(f"Training AutoGluon TimeSeries (time_limit={time_limit}s, preset={presets})...")
        self.predictor = TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            target=self.target_col,
            eval_metric="MAE",
            path=self.path,
            freq="h",  # explicitly specify hourly frequency
        )
        
        self.predictor.fit(
            train_data,
            presets=presets,
            time_limit=time_limit,
            random_seed=42
        )
        logger.info("AutoGluon TimeSeries training complete.")
        return self

    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        """Forecast the OOS period using the fitted predictor.
        
        AutoGluon TimeSeries forecasts `prediction_length` steps into the future 
        immediately following the end of the `train_df`.
        We must provide the `known_covariates` for the future period via `test_df`.
        """
        if self.predictor is None:
            raise ValueError("Predictor not fitted. Call fit() first.")
            
        train_data = self._prepare_ts_data(train_df)
            
        logger.info("Generating AutoGluon TimeSeries predictions...")
        predictions = self.predictor.predict(
            train_data,
            random_seed=42
        )
        
        # Predictions return a dataframe structured as (item_id, timestamp) index
        # Extract the 'mean' column and map to the original timestamp index
        preds_series = predictions.reset_index().set_index("timestamp")["mean"]
        
        # Ensure the timezone aligns (AutoGluon might drop it during intermediate steps, though mostly preserved)
        if preds_series.index.tz is None and test_df.index.tz is not None:
             preds_series.index = preds_series.index.tz_localize("UTC")
             
        # Realign strictly to test_df index
        return preds_series.reindex(test_df.index)

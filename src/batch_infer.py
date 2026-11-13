"""
Cloud-Based ML Sales Forecasting
Azure ML - Batch Inference

Loads a registered LightGBM model and its
preprocessing artifacts to produce 7-day
sales forecasts.

No model training or evaluation is performed here.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FORECAST_ROWS = 178_773
EXPECTED_SERIES = 25_539
FORECAST_HORIZON = 7

FINAL_ORIGIN = pd.Timestamp("2024-06-25")

FORECAST_START = pd.Timestamp("2024-06-26")
FORECAST_END = pd.Timestamp("2024-07-02")

KEY_COLUMNS = [
    "store_id",
    "product_id",
    "target_date",
]

CATEGORICAL_FEATURES = [
    "store_id",
    "city_id",
    "product_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "day_of_week",
]


# ============================================================
# LOAD REGISTERED MODEL
# ============================================================

def load_model_artifacts(model_dir):
    model_dir = Path(model_dir)

    model_path = (
        model_dir / "lightgbm_final.joblib"
    )

    preprocessing_path = (
        model_dir / "preprocessing.joblib"
    )

    for path in [
        model_path,
        preprocessing_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {path}"
            )

    print("Loading registered LightGBM model...")

    model = joblib.load(model_path)

    preprocessing = joblib.load(
        preprocessing_path
    )

    features = preprocessing["features"]
    category_levels = preprocessing["categories"]

    if len(features) != 23:
        raise ValueError(
            f"Expected 23 features, found {len(features)}"
        )

    for col in CATEGORICAL_FEATURES:
        if col not in category_levels:
            raise ValueError(
                f"Missing categorical metadata: {col}"
            )

    print("Model artifacts loaded successfully.")

    return model, features, category_levels


# ============================================================
# LOAD FORECAST DATA
# ============================================================

def load_forecast_data(input_path, features):
    input_path = Path(input_path)

    if input_path.is_dir():
        input_path = (
            input_path / "final_forecast_features.parquet"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Forecast input not found: {input_path}"
        )

    print("Loading forecast features...")

    forecast = pd.read_parquet(input_path)

    if len(forecast) != EXPECTED_FORECAST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_FORECAST_ROWS:,} rows, "
            f"found {len(forecast):,}"
        )

    required = (
        features + KEY_COLUMNS + ["origin"]
    )

    missing = [
        col for col in required
        if col not in forecast.columns
    ]

    if missing:
        raise ValueError(
            f"Missing forecast columns: {missing}"
        )

    if "target_sale" in forecast.columns:
        raise ValueError(
            "Data leakage: actual future sales "
            "must not be present in forecast input."
        )

    forecast["origin"] = pd.to_datetime(
        forecast["origin"]
    )

    forecast["target_date"] = pd.to_datetime(
        forecast["target_date"]
    )

    if not (
        forecast["origin"] == FINAL_ORIGIN
    ).all():
        raise ValueError(
            "Unexpected forecast origin."
        )

    if not forecast["target_date"].between(
        FORECAST_START,
        FORECAST_END,
    ).all():
        raise ValueError(
            "Unexpected forecast target dates."
        )

    if forecast.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Duplicate forecast records."
        )

    series_counts = (
        forecast.groupby(
            ["store_id", "product_id"]
        )["target_date"]
        .nunique()
    )

    if (
        len(series_counts) != EXPECTED_SERIES
        or not series_counts.eq(FORECAST_HORIZON).all()
    ):
        raise ValueError(
            "Incomplete 7-day forecasting coverage."
        )

    if forecast[features].isna().any().any():
        raise ValueError(
            "Missing feature values detected."
        )

    print(f"Forecast rows: {len(forecast):,}")
    print(f"Store-product series: {len(series_counts):,}")
    print(f"Forecast horizon: {FORECAST_HORIZON} days")

    return forecast


# ============================================================
# RESTORE CATEGORICAL ENCODING
# ============================================================

def apply_preprocessing(
    forecast,
    category_levels,
):
    """
    Apply the same categorical feature definitions
    used during model training.
    """

    for col in CATEGORICAL_FEATURES:
        dtype = pd.CategoricalDtype(
            categories=category_levels[col]
        )

        forecast[col] = forecast[col].astype(dtype)

        if forecast[col].isna().any():
            raise ValueError(
                f"Unknown category detected in {col}"
            )

    return forecast


# ============================================================
# GENERATE FORECASTS
# ============================================================

def generate_predictions(
    model,
    forecast,
    features,
):
    print("\n=== CLOUD BATCH INFERENCE ===")

    predictions = model.predict(
        forecast[features]
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Non-finite model predictions detected."
        )

    # Sales predictions cannot be negative.
    predictions = np.maximum(
        predictions,
        0.0,
    )

    result = forecast[
        KEY_COLUMNS
    ].copy()

    result["predicted_sale"] = predictions

    result = (
        result.sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    print(f"Predictions generated: {len(result):,}")

    return result


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_predictions(
    predictions,
    output_dir,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "final_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print("\n=== BATCH INFERENCE COMPLETED ===")
    print(f"Saved: {output_path}")
    print(f"Rows: {len(predictions):,}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Registered Azure ML model directory.",
    )

    parser.add_argument(
        "--forecast-data",
        required=True,
        help="Forecast features input.",
    )

    parser.add_argument(
        "--predictions-output",
        required=True,
        help="Azure ML predictions output directory.",
    )

    args = parser.parse_args()

    model, features, category_levels = (
        load_model_artifacts(args.model_dir)
    )

    forecast = load_forecast_data(
        args.forecast_data,
        features,
    )

    forecast = apply_preprocessing(
        forecast,
        category_levels,
    )

    predictions = generate_predictions(
        model,
        forecast,
        features,
    )

    save_predictions(
        predictions,
        args.predictions_output,
    )


if __name__ == "__main__":
    main()

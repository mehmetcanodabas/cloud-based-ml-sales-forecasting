"""
Cloud-Based ML Sales Forecasting
Azure ML - LightGBM Training

Trains the final 7-day forecasting model using
historical supervised training examples.

No validation or hyperparameter tuning is performed.
Final evaluation sales are not used for training.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FINAL_TRAINING_CUTOFF = pd.Timestamp("2024-06-25")

EXPECTED_TRAINING_ROWS = 1_608_957
EXPECTED_FEATURE_COUNT = 23

EXCLUDED_COLUMNS = [
    "origin",
    "target_date",
    "target_sale",
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
# INPUT VALIDATION
# ============================================================

def load_training_data(input_path):
    input_path = Path(input_path)

    if input_path.is_dir():
        input_path = (
            input_path / "training_features.parquet"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {input_path}"
        )

    print("Loading training features...", flush=True)

    df = pd.read_parquet(input_path)

    if len(df) != EXPECTED_TRAINING_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_ROWS:,} rows, "
            f"found {len(df):,}"
        )

    required_columns = (
        EXCLUDED_COLUMNS + CATEGORICAL_FEATURES
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing training columns: {missing}"
        )

    features = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(features)}"
        )

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )

    if not (
        df["target_date"] <= FINAL_TRAINING_CUTOFF
    ).all():
        raise ValueError(
            "Data leakage: future sales targets "
            "found in training data."
        )

    if df[features + ["target_sale"]].isna().any().any():
        raise ValueError(
            "Missing values found in training data."
        )

    if not pd.api.types.is_numeric_dtype(
        df["target_sale"]
    ):
        raise ValueError(
            "Training target must be numeric."
        )

    print(f"Training rows: {len(df):,}")
    print(f"Model features: {len(features)}")
    print(
        "Latest training target date:",
        df["target_date"].max().date(),
    )

    return df, features


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

def encode_categories(df):
    """
    Preserve categorical encoding information so
    batch inference uses identical categories.
    """

    category_levels = {}

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

        category_levels[col] = (
            df[col].cat.categories.tolist()
        )

    return df, category_levels


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(df, features):
    print("\n=== CLOUD MODEL TRAINING ===", flush=True)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        n_jobs=6,
        random_state=42,
        verbosity=-1,
    )

    start_time = time.perf_counter()

    model.fit(
        df[features],
        df["target_sale"],
        categorical_feature=CATEGORICAL_FEATURES,
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"Training completed in {elapsed:.2f} seconds",
        flush=True,
    )

    return model, elapsed


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    features,
    category_levels,
    elapsed,
    output_dir,
    training_rows,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        output_dir / "lightgbm_final.joblib"
    )

    preprocessing_path = (
        output_dir / "preprocessing.joblib"
    )

    metadata_path = (
        output_dir / "training_metadata.json"
    )

    joblib.dump(model, model_path)

    joblib.dump(
        {
            "features": features,
            "categories": category_levels,
        },
        preprocessing_path,
    )

    metadata = {
        "algorithm": "LightGBM Regression",
        "forecast_horizon_days": 7,
        "training_cutoff": "2024-06-25",
        "training_rows": training_rows,
        "feature_count": len(features),
        "training_seconds": elapsed,
        "model_parameters": model.get_params(),
        "feature_names": features,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n=== MODEL ARTIFACTS SAVED ===")
    print(model_path)
    print(preprocessing_path)
    print(metadata_path)


# ============================================================
# AZURE ML METRICS
# ============================================================

def log_training_metrics(
    training_rows,
    feature_count,
    elapsed,
):
    try:
        import mlflow

        mlflow.log_metric(
            "training_rows",
            training_rows,
        )

        mlflow.log_metric(
            "feature_count",
            feature_count,
        )

        mlflow.log_metric(
            "training_seconds",
            elapsed,
        )

        print("Training metrics logged to MLflow.")

    except ImportError:
        print(
            "MLflow is unavailable. "
            "Training metadata was saved locally."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-data",
        required=True,
        help="Training Parquet file or input directory.",
    )

    parser.add_argument(
        "--model-output",
        required=True,
        help="Azure ML model output directory.",
    )

    args = parser.parse_args()

    df, features = load_training_data(
        args.training_data
    )

    df, category_levels = encode_categories(df)

    model, elapsed = train_model(
        df,
        features,
    )

    save_artifacts(
        model=model,
        features=features,
        category_levels=category_levels,
        elapsed=elapsed,
        output_dir=args.model_output,
        training_rows=len(df),
    )

    log_training_metrics(
        training_rows=len(df),
        feature_count=len(features),
        elapsed=elapsed,
    )

    print(
        "\nAzure ML training script completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()

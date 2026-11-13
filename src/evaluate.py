"""
Cloud-Based ML Sales Forecasting
Azure ML - Final Evaluation

Evaluates the held-out seven-day forecast period:
2024-06-26 to 2024-07-02.

Actual future sales are used only for evaluation,
never for training or batch inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_START = pd.Timestamp("2024-06-26")
FORECAST_END = pd.Timestamp("2024-07-02")

EXPECTED_SERIES = 25_539
FORECAST_HORIZON = 7
EXPECTED_ROWS = EXPECTED_SERIES * FORECAST_HORIZON

KEY_COLUMNS = [
    "store_id",
    "product_id",
    "target_date",
]


def find_predictions(path):
    path = Path(path)

    if path.is_dir():
        path = path / "final_predictions.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Predictions not found: {path}"
        )

    return path


def load_predictions(path):
    predictions = pd.read_csv(
        find_predictions(path)
    )

    required = KEY_COLUMNS + ["predicted_sale"]

    missing = [
        column for column in required
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"]
    )

    if len(predictions) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} predictions, "
            f"found {len(predictions):,}"
        )

    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Duplicate prediction keys detected."
        )

    if not predictions["target_date"].between(
        FORECAST_START,
        FORECAST_END,
    ).all():
        raise ValueError(
            "Unexpected forecast dates."
        )

    if not np.isfinite(
        predictions["predicted_sale"].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite predictions detected."
        )

    return predictions


def load_actuals(path, selected_pairs):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation data not found: {path}"
        )

    actuals = pd.read_parquet(
        path,
        columns=[
            "store_id",
            "product_id",
            "dt",
            "sale_amount",
        ],
    )

    actuals = actuals.rename(
        columns={
            "dt": "target_date",
            "sale_amount": "actual_sale",
        }
    )

    actuals["target_date"] = pd.to_datetime(
        actuals["target_date"]
    )

    actuals = actuals.loc[
        actuals["target_date"].between(
            FORECAST_START,
            FORECAST_END,
        )
    ]

    actuals = actuals.merge(
        selected_pairs,
        on=["store_id", "product_id"],
        how="inner",
        validate="many_to_one",
    )

    if actuals.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Duplicate actual sales records."
        )

    return actuals


def calculate_wape(actual, predicted):
    denominator = np.abs(actual).sum()

    if denominator <= 0:
        raise ValueError(
            "WAPE denominator must be positive."
        )

    return (
        np.abs(actual - predicted).sum()
        / denominator
        * 100.0
    )


def evaluate(predictions, actuals):
    merged = predictions.merge(
        actuals,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected evaluation row count."
        )

    if merged["actual_sale"].isna().any():
        raise ValueError(
            "Some forecast records have no actual sales."
        )

    actual = merged["actual_sale"].to_numpy(
        dtype=np.float64
    )

    predicted = merged["predicted_sale"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(actual).all():
        raise ValueError(
            "Invalid actual sales detected."
        )

    daily_mae = float(
        np.mean(np.abs(actual - predicted))
    )

    daily_wape = float(
        calculate_wape(actual, predicted)
    )

    # Aggregate the seven forecast days
    # for each product across all selected stores.
    weekly_products = (
        merged.groupby(
            "product_id",
            as_index=False,
        )[["actual_sale", "predicted_sale"]]
        .sum()
    )

    weekly_product_wape = float(
        calculate_wape(
            weekly_products["actual_sale"].to_numpy(),
            weekly_products["predicted_sale"].to_numpy(),
        )
    )

    metrics = {
        "evaluation_start": str(
            FORECAST_START.date()
        ),
        "evaluation_end": str(
            FORECAST_END.date()
        ),
        "forecast_horizon_days": FORECAST_HORIZON,
        "forecast_rows": len(merged),
        "store_product_series": EXPECTED_SERIES,
        "evaluated_products": int(
            weekly_products["product_id"].nunique()
        ),
        "daily_mae": daily_mae,
        "daily_wape_percent": daily_wape,
        "weekly_product_wape_percent": weekly_product_wape,
    }

    return metrics


def save_metrics(metrics, output_path):
    output_path = Path(output_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_file = (
        output_path / "evaluation_metrics.json"
    )

    metrics_file.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL EVALUATION ===")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nSaved: {metrics_file}")

    try:
        import mlflow

        for key in [
            "daily_mae",
            "daily_wape_percent",
            "weekly_product_wape_percent",
        ]:
            mlflow.log_metric(
                key,
                metrics[key],
            )

        print("Evaluation metrics logged to MLflow.")

    except ImportError:
        print(
            "MLflow unavailable; JSON metrics saved."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
    )

    parser.add_argument(
        "--actual-data",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    predictions = load_predictions(
        args.predictions
    )

    selected_pairs = (
        predictions[
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
    )

    actuals = load_actuals(
        args.actual_data,
        selected_pairs,
    )

    metrics = evaluate(
        predictions,
        actuals,
    )

    save_metrics(
        metrics,
        args.output,
    )


if __name__ == "__main__":
    main()

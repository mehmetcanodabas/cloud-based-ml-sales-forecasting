"""
Retail Sales Forecasting - Feature Engineering

Commands:
    python build_features.py train
    python build_features.py forecast

Model:
    LightGBM with 23 features

Important:
    - Historical sales and stock features use only
      data available at the forecast origin.
    - Future weather, discounts and activities are
      treated as supplied scenario inputs.
    - Final evaluation sales are never read during
      forecast feature generation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HISTORY_PATH = DATA_DIR / "train_selected.parquet"
EVAL_PATH = DATA_DIR / "eval.parquet"

CANDIDATE_PATH = (
    DATA_DIR / "training_features_candidate.parquet"
)

FORECAST_PATH = (
    DATA_DIR / "final_forecast_features.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = pd.Timestamp("2024-03-28")
FINAL_ORIGIN = pd.Timestamp("2024-06-25")

DAYS_PER_SERIES = 90
FORECAST_HORIZON = 7
EXPECTED_SERIES = 25_539

ORIGINS = [
    "2024-04-24",
    "2024-05-01",
    "2024-05-08",
    "2024-05-15",
    "2024-05-22",
    "2024-05-29",
    "2024-06-04",
    "2024-06-11",
    "2024-06-18",
]

SERIES_KEYS = [
    "store_id",
    "product_id",
]

ID_FEATURES = [
    "store_id",
    "city_id",
    "product_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
]

MODEL_FEATURES = ID_FEATURES + [
    "horizon",
    "day_of_week",
    "last_sale",
    "mean_7",
    "mean_14",
    "mean_28",
    "same_weekday_last_week",
    "stockout_hours_7",
    "temperature",
    "precipitation",
    "discount",
    "activity_flag",
    "holiday_flag",
    "product_mean_7_all_stores",
    "store_category_mean_7",
    "discount_change",
]

OUTPUT_COLUMNS = (
    ID_FEATURES
    + ["origin", "target_date"]
    + MODEL_FEATURES[len(ID_FEATURES):]
)

# Include the weather, discount and activity
# columns needed when constructing training features.
HISTORY_COLUMNS = ID_FEATURES + [
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "avg_temperature",
    "precpt",
    "activity_flag",
    "holiday_flag",
]

# Final evaluation: only scenario inputs.
# Never load sale_amount or future stock information.
SCENARIO_COLUMNS = SERIES_KEYS + [
    "dt",
    "avg_temperature",
    "precpt",
    "discount",
    "activity_flag",
    "holiday_flag",
]


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_history():
    print("Loading historical sales...")

    daily = pd.read_parquet(
        HISTORY_PATH,
        columns=HISTORY_COLUMNS,
    )

    daily["dt"] = pd.to_datetime(daily["dt"])

    daily = daily.sort_values(
        SERIES_KEYS + ["dt"]
    ).reset_index(drop=True)

    n_series = (
        daily[SERIES_KEYS]
        .drop_duplicates()
        .shape[0]
    )

    expected_rows = (
        EXPECTED_SERIES * DAYS_PER_SERIES
    )

    if n_series != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES:,} series, "
            f"found {n_series:,}."
        )

    if len(daily) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {len(daily):,}."
        )

    if daily.duplicated(
        SERIES_KEYS + ["dt"]
    ).any():
        raise ValueError(
            "Duplicate store-product-date records."
        )

    actual_dates = (
        daily["dt"]
        .to_numpy()
        .reshape(n_series, DAYS_PER_SERIES)
    )

    expected_dates = pd.date_range(
        START_DATE,
        periods=DAYS_PER_SERIES,
        freq="D",
    ).to_numpy()

    if not np.array_equal(
        actual_dates,
        np.broadcast_to(
            expected_dates,
            actual_dates.shape,
        ),
    ):
        raise ValueError(
            "Historical dates are not continuous."
        )

    print(f"Historical series: {n_series:,}")
    print(f"Historical rows: {len(daily):,}")

    return daily, n_series


# ============================================================
# HISTORICAL FEATURES
# ============================================================

def calculate_origin_features(
    daily,
    n_series,
    origin,
):
    """
    Calculate features using only sales and stock
    observed on or before the forecast origin.
    """

    origin_idx = (
        origin - START_DATE
    ).days

    if not (
        27 <= origin_idx < DAYS_PER_SERIES
    ):
        raise ValueError(
            f"Invalid forecast origin: {origin.date()}"
        )

    sales = (
        daily["sale_amount"]
        .to_numpy()
        .reshape(n_series, DAYS_PER_SERIES)
    )

    stock = (
        daily["stock_hour6_22_cnt"]
        .to_numpy()
        .reshape(n_series, DAYS_PER_SERIES)
    )

    row_indices = (
        np.arange(n_series) * DAYS_PER_SERIES
        + origin_idx
    )

    origin_rows = (
        daily.iloc[row_indices]
        .reset_index(drop=True)
    )

    # ----------------------------------------
    # Historical sales and stock
    # ----------------------------------------

    values = {
        "last_sale": sales[:, origin_idx],

        "mean_7": sales[
            :, origin_idx - 6:origin_idx + 1
        ].mean(axis=1),

        "mean_14": sales[
            :, origin_idx - 13:origin_idx + 1
        ].mean(axis=1),

        "mean_28": sales[
            :, origin_idx - 27:origin_idx + 1
        ].mean(axis=1),

        "stockout_hours_7": stock[
            :, origin_idx - 6:origin_idx + 1
        ].sum(axis=1),

        "discount_at_origin": (
            origin_rows["discount"].to_numpy()
        ),

        # For target horizons 1-7, these are
        # the sales from target_date minus 7 days.
        "same_weekday_last_week": sales[
            :, origin_idx - 6:origin_idx + 1
        ],
    }

    # ----------------------------------------
    # Product mean across selected stores
    # ----------------------------------------

    week_start = (
        origin - pd.Timedelta(days=6)
    )

    history_week = daily.loc[
        daily["dt"].between(
            week_start,
            origin,
        )
    ]

    product_daily = (
        history_week.groupby(
            ["product_id", "dt"],
            sort=False,
        )["sale_amount"]
        .mean()
    )

    product_mean = (
        product_daily
        .groupby(level=0)
        .mean()
    )

    values["product_mean_7_all_stores"] = (
        origin_rows["product_id"]
        .map(product_mean)
        .to_numpy()
    )

    # ----------------------------------------
    # Store-category historical mean
    # ----------------------------------------

    category_keys = [
        "store_id",
        "first_category_id",
    ]

    category_daily = (
        history_week.groupby(
            category_keys + ["dt"],
            sort=False,
        )["sale_amount"]
        .mean()
    )

    category_mean = (
        category_daily.groupby(
            level=[0, 1]
        ).mean()
    )

    category_index = (
        pd.MultiIndex.from_frame(
            origin_rows[category_keys]
        )
    )

    values["store_category_mean_7"] = (
        category_mean
        .reindex(category_index)
        .to_numpy()
    )

    for feature in [
        "product_mean_7_all_stores",
        "store_category_mean_7",
    ]:
        if not np.isfinite(
            values[feature]
        ).all():
            raise ValueError(
                f"Invalid historical feature: {feature}"
            )

    return origin_rows, values


# ============================================================
# CHECK SCENARIO ORDER
# ============================================================

def align_scenario(
    scenario,
    origin_rows,
    origin,
):
    """
    Require exactly seven future dates for
    each selected store-product series.
    """

    scenario = scenario.sort_values(
        SERIES_KEYS + ["dt"]
    ).reset_index(drop=True)

    n_series = len(origin_rows)

    expected_store = np.repeat(
        origin_rows["store_id"].to_numpy(),
        FORECAST_HORIZON,
    )

    expected_product = np.repeat(
        origin_rows["product_id"].to_numpy(),
        FORECAST_HORIZON,
    )

    expected_dates = np.tile(
        pd.date_range(
            origin + pd.Timedelta(days=1),
            periods=FORECAST_HORIZON,
        ).to_numpy(),
        n_series,
    )

    if len(scenario) != len(expected_dates):
        raise ValueError(
            "Unexpected number of scenario rows."
        )

    if not np.array_equal(
        scenario["store_id"].to_numpy(),
        expected_store,
    ):
        raise ValueError(
            "Scenario store IDs do not match."
        )

    if not np.array_equal(
        scenario["product_id"].to_numpy(),
        expected_product,
    ):
        raise ValueError(
            "Scenario product IDs do not match."
        )

    if not np.array_equal(
        scenario["dt"].to_numpy(),
        expected_dates,
    ):
        raise ValueError(
            "Scenario dates do not match."
        )

    return scenario


# ============================================================
# BUILD FEATURE ROWS
# ============================================================

def build_rows(
    origin_rows,
    values,
    origin,
    scenario,
    include_target,
):
    n_series = len(origin_rows)

    n_rows = (
        n_series * FORECAST_HORIZON
    )

    horizons = np.tile(
        np.arange(1, FORECAST_HORIZON + 1),
        n_series,
    )

    target_dates = np.tile(
        pd.date_range(
            origin + pd.Timedelta(days=1),
            periods=FORECAST_HORIZON,
        ).to_numpy(),
        n_series,
    )

    result = {}

    # ----------------------------------------
    # Identifiers
    # ----------------------------------------

    for col in ID_FEATURES:
        result[col] = np.repeat(
            origin_rows[col].to_numpy(),
            FORECAST_HORIZON,
        )

    result["origin"] = np.repeat(
        origin.strftime("%Y-%m-%d"),
        n_rows,
    )

    result["target_date"] = (
        pd.DatetimeIndex(target_dates)
        .strftime("%Y-%m-%d")
    )

    result["horizon"] = horizons

    result["day_of_week"] = (
        pd.DatetimeIndex(target_dates)
        .day_of_week
        .to_numpy()
    )

    # ----------------------------------------
    # Historical features
    # ----------------------------------------

    fixed_features = [
        "last_sale",
        "mean_7",
        "mean_14",
        "mean_28",
        "stockout_hours_7",
        "product_mean_7_all_stores",
        "store_category_mean_7",
    ]

    for col in fixed_features:
        result[col] = np.repeat(
            values[col],
            FORECAST_HORIZON,
        )

    result["same_weekday_last_week"] = (
        values["same_weekday_last_week"]
        .reshape(-1)
    )

    # ----------------------------------------
    # Future scenario features
    # ----------------------------------------

    result["temperature"] = (
        scenario["avg_temperature"].to_numpy()
    )

    result["precipitation"] = (
        scenario["precpt"].to_numpy()
    )

    result["discount"] = (
        scenario["discount"].to_numpy()
    )

    result["activity_flag"] = (
        scenario["activity_flag"].to_numpy()
    )

    result["holiday_flag"] = (
        scenario["holiday_flag"].to_numpy()
    )

    result["discount_change"] = (
        result["discount"]
        - np.repeat(
            values["discount_at_origin"],
            FORECAST_HORIZON,
        )
    )

    if include_target:
        result["target_sale"] = (
            scenario["sale_amount"].to_numpy()
        )

    frame = pd.DataFrame(result)

    columns = OUTPUT_COLUMNS.copy()

    if include_target:
        columns.append("target_sale")

    frame = frame[columns]

    if len(frame) != n_rows:
        raise ValueError(
            "Unexpected generated feature row count."
        )

    if len(MODEL_FEATURES) != 23:
        raise ValueError(
            "Expected exactly 23 model features."
        )

    if frame[MODEL_FEATURES].isna().any().any():
        raise ValueError(
            "Missing values in generated features."
        )

    return frame


# ============================================================
# TRAIN MODE
# ============================================================

def generate_train():
    print("\n=== GENERATING TRAINING FEATURES ===")

    if CANDIDATE_PATH.exists():
        raise FileExistsError(
            f"Candidate already exists: {CANDIDATE_PATH}"
        )

    daily, n_series = load_history()

    writer = None
    total_rows = 0

    try:
        for origin_text in ORIGINS:
            origin = pd.Timestamp(
                origin_text
            )

            origin_rows, values = (
                calculate_origin_features(
                    daily,
                    n_series,
                    origin,
                )
            )

            scenario = daily.loc[
                daily["dt"].between(
                    origin + pd.Timedelta(days=1),
                    origin + pd.Timedelta(
                        days=FORECAST_HORIZON
                    ),
                )
            ]

            scenario = align_scenario(
                scenario,
                origin_rows,
                origin,
            )

            frame = build_rows(
                origin_rows,
                values,
                origin,
                scenario,
                include_target=True,
            )

            table = pa.Table.from_pandas(
                frame,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    CANDIDATE_PATH,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

            total_rows += len(frame)

            print(
                f"{origin_text}: "
                f"{len(frame):,} rows"
            )

    finally:
        if writer is not None:
            writer.close()

    expected_rows = (
        EXPECTED_SERIES
        * len(ORIGINS)
        * FORECAST_HORIZON
    )

    if total_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"got {total_rows:,}"
        )

    print("\n=== TRAINING FEATURES CREATED ===")
    print(f"Rows: {total_rows:,}")
    print(f"Model features: {len(MODEL_FEATURES)}")
    print(f"Total columns: {len(OUTPUT_COLUMNS) + 1}")
    print(f"Saved: {CANDIDATE_PATH}")


# ============================================================
# FINAL FORECAST MODE
# ============================================================

def generate_forecast():
    print("\n=== FINAL FORECAST FEATURES ===")

    if FORECAST_PATH.exists():
        raise FileExistsError(
            f"Forecast file already exists: "
            f"{FORECAST_PATH}"
        )

    daily, n_series = load_history()

    origin_rows, values = (
        calculate_origin_features(
            daily,
            n_series,
            FINAL_ORIGIN,
        )
    )

    # Do not read future sales or future stock.
    scenario = pd.read_parquet(
        EVAL_PATH,
        columns=SCENARIO_COLUMNS,
    )

    scenario["dt"] = pd.to_datetime(
        scenario["dt"]
    )

    selected_pairs = (
        origin_rows[SERIES_KEYS]
        .drop_duplicates()
    )

    scenario = scenario.merge(
        selected_pairs,
        on=SERIES_KEYS,
        how="inner",
        validate="many_to_one",
    )

    scenario = align_scenario(
        scenario,
        origin_rows,
        FINAL_ORIGIN,
    )

    frame = build_rows(
        origin_rows,
        values,
        FINAL_ORIGIN,
        scenario,
        include_target=False,
    )

    expected_rows = (
        EXPECTED_SERIES * FORECAST_HORIZON
    )

    if len(frame) != expected_rows:
        raise ValueError(
            "Incorrect final forecast row count."
        )

    if "target_sale" in frame.columns:
        raise ValueError(
            "Final forecast must not contain targets."
        )

    frame.to_parquet(
        FORECAST_PATH,
        index=False,
        compression="snappy",
    )

    print("\n=== FORECAST FEATURES CREATED ===")
    print(f"Series: {n_series:,}")
    print(f"Forecast days: {FORECAST_HORIZON}")
    print(f"Rows: {len(frame):,}")
    print(f"Model features: {len(MODEL_FEATURES)}")
    print(f"Saved: {FORECAST_PATH}")

    print("\nFinal sales were not accessed.")

    print(
        "Future weather, discounts and activity "
        "were used as supplied scenario inputs."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build retail sales forecasting features."
        )
    )

    parser.add_argument(
        "mode",
        choices=[
            "train",
            "forecast",
        ],
        help="Feature generation mode.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        generate_train()

    elif args.mode == "forecast":
        generate_forecast()


if __name__ == "__main__":
    main()

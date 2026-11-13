"""
Cloud-Based ML Sales Forecasting
Local visualization of Azure ML forecasting results.

Outputs (docs/results/):
- five_store_final_summary.png
- store_<ID>_top30_products.png (5 files)
- selected_store_metrics.csv

Uses Azure ML predictions and FreshRetailNet data.
Does not submit Azure jobs or retrain the model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_DIR = ROOT / "evidence" / "inference"
TRAIN_PATH = ROOT / "raw_data" / "train.parquet"
EVAL_PATH = ROOT / "raw_data" / "eval.parquet"

OUTPUT_DIR = ROOT / "docs" / "results"

FORECAST_START = pd.Timestamp("2024-06-26")
FORECAST_END = pd.Timestamp("2024-07-02")

HISTORY_START = pd.Timestamp("2024-06-19")
HISTORY_END = pd.Timestamp("2024-06-25")

EXPECTED_ROWS = 178_773

N_STORES = 5
N_SELECTED_PRODUCTS = 50
N_DISPLAY_PRODUCTS = 30

PRIORITY_THRESHOLD = 5.0

ACTUAL_COLOR = "#568DBB"
PREDICTED_COLOR = "#2C9275"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})


# ============================================================
# LOAD AZURE PREDICTIONS
# ============================================================

def find_predictions():
    files = list(
        PREDICTIONS_DIR.rglob("final_predictions.csv")
    )

    if len(files) != 1:
        raise FileNotFoundError(
            "Expected exactly one final_predictions.csv "
            f"inside {PREDICTIONS_DIR}"
        )

    return files[0]


def load_data():
    print("\n=== LOADING AZURE FORECAST RESULTS ===")

    prediction_path = find_predictions()

    predictions = pd.read_csv(prediction_path)

    required = [
        "store_id",
        "product_id",
        "target_date",
        "predicted_sale",
    ]

    missing = [
        col for col in required
        if col not in predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"]
    )

    actuals = pd.read_parquet(
        EVAL_PATH,
        columns=[
            "store_id",
            "product_id",
            "dt",
            "sale_amount",
        ],
    )

    actuals = actuals.rename(columns={
        "dt": "target_date",
        "sale_amount": "actual_sale",
    })

    actuals["target_date"] = pd.to_datetime(
        actuals["target_date"]
    )

    actuals = actuals.loc[
        actuals["target_date"].between(
            FORECAST_START,
            FORECAST_END,
        )
    ]

    keys = [
        "store_id",
        "product_id",
        "target_date",
    ]

    if predictions.duplicated(keys).any():
        raise ValueError(
            "Duplicate prediction records."
        )

    if actuals.duplicated(keys).any():
        raise ValueError(
            "Duplicate actual sales records."
        )

    merged = predictions.merge(
        actuals,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(merged):,}."
        )

    if merged["actual_sale"].isna().any():
        raise ValueError(
            "Some predictions have no actual sales."
        )

    if not np.isfinite(
        merged[
            ["actual_sale", "predicted_sale"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid actual or predicted values."
        )

    if not merged["target_date"].between(
        FORECAST_START,
        FORECAST_END,
    ).all():
        raise ValueError(
            "Unexpected forecast evaluation dates."
        )

    print(f"Forecast rows: {len(merged):,}")
    print(f"Unique stores: {merged['store_id'].nunique():,}")
    print(f"Unique products: {merged['product_id'].nunique():,}")

    print("\nLoading historical sales...")

    history = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "store_id",
            "product_id",
            "dt",
            "sale_amount",
        ],
    )

    history["dt"] = pd.to_datetime(
        history["dt"]
    )

    history = history.loc[
        history["dt"].between(
            HISTORY_START,
            HISTORY_END,
        )
    ]

    print(f"Historical rows: {len(history):,}")

    return merged, history


# ============================================================
# ERROR METRICS
# ============================================================

def wape(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    denominator = np.abs(actual).sum()

    if denominator <= 0:
        return np.nan

    return float(
        np.abs(actual - predicted).sum()
        / denominator
        * 100.0
    )


# ============================================================
# SELECT 50 PRODUCTS PER STORE
# ============================================================

def build_store_product_data(merged, history):
    print("\n=== SELECTING PRODUCTS PER STORE ===")

    forecast_pairs = (
        merged[
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
    )

    history = history.merge(
        forecast_pairs,
        on=["store_id", "product_id"],
        how="inner",
        validate="many_to_one",
    )

    historical_products = (
        history.groupby(
            ["store_id", "product_id"],
            as_index=False,
        )["sale_amount"]
        .sum()
        .rename(columns={
            "sale_amount": "historical_weekly_sales"
        })
    )

    historical_products = historical_products.sort_values(
        [
            "store_id",
            "historical_weekly_sales",
            "product_id",
        ],
        ascending=[True, False, True],
    )

    selected_products = (
        historical_products
        .groupby(
            "store_id",
            group_keys=False,
        )
        .head(N_SELECTED_PRODUCTS)
        .copy()
    )

    product_counts = (
        selected_products.groupby("store_id")
        .size()
    )

    eligible_store_ids = product_counts[
        product_counts == N_SELECTED_PRODUCTS
    ].index

    selected_products = selected_products.loc[
        selected_products["store_id"].isin(
            eligible_store_ids
        )
    ]

    weekly_forecasts = (
        merged.groupby(
            ["store_id", "product_id"],
            as_index=False,
        )[["actual_sale", "predicted_sale"]]
        .sum()
    )

    selected = selected_products.merge(
        weekly_forecasts,
        on=["store_id", "product_id"],
        how="inner",
        validate="one_to_one",
    )

    print(
        "Stores with 50 selected products:",
        selected["store_id"].nunique(),
    )

    return selected


# ============================================================
# RANK STORES BY WEEKLY WAPE
# ============================================================

def select_best_stores(selected, merged):
    print("\n=== RANKING STORES ===")

    rows = []

    # Total weekly actual sales of forecasted products
    # in each store. Used for sales coverage.
    all_store_actuals = (
        merged.groupby("store_id")["actual_sale"]
        .sum()
    )

    for store_id, products in selected.groupby("store_id"):

        if len(products) != N_SELECTED_PRODUCTS:
            continue

        selected_pairs = products[
            ["store_id", "product_id"]
        ]

        store_daily = merged.merge(
            selected_pairs,
            on=["store_id", "product_id"],
            how="inner",
            validate="many_to_one",
        )

        daily_wape = wape(
            store_daily["actual_sale"],
            store_daily["predicted_sale"],
        )

        weekly_wape = wape(
            products["actual_sale"],
            products["predicted_sale"],
        )

        historical_volume = float(
            products["historical_weekly_sales"].sum()
        )

        priority_count = int(
            (
                products["historical_weekly_sales"]
                >= PRIORITY_THRESHOLD
            ).sum()
        )

        total_store_sales = float(
            all_store_actuals.loc[store_id]
        )

        if total_store_sales > 0:
            coverage = (
                products["actual_sale"].sum()
                / total_store_sales
                * 100.0
            )
        else:
            coverage = np.nan

        rows.append({
            "store_id": store_id,
            "daily_wape": daily_wape,
            "weekly_wape": weekly_wape,
            "priority_count": priority_count,
            "priority_total": N_SELECTED_PRODUCTS,
            "sales_coverage": coverage,
            "historical_volume": historical_volume,
        })

    results = pd.DataFrame(rows)

    results = results.dropna(
        subset=[
            "daily_wape",
            "weekly_wape",
            "sales_coverage",
        ]
    )

    if len(results) < N_STORES:
        raise ValueError(
            "Not enough eligible stores."
        )

    # Avoid ranking only very low-volume stores.
    # This threshold uses pre-forecast history.
    minimum_volume = results[
        "historical_volume"
    ].median()

    eligible = results.loc[
        results["historical_volume"]
        >= minimum_volume
    ].copy()

    if len(eligible) < N_STORES:
        raise ValueError(
            "Not enough eligible stores after "
            "historical volume filtering."
        )

    best = (
        eligible.sort_values(
            ["weekly_wape", "store_id"],
            ascending=[True, True],
        )
        .head(N_STORES)
        .reset_index(drop=True)
    )

    print("\n=== FIVE BEST ELIGIBLE STORES ===")

    print(
        best[
            [
                "store_id",
                "daily_wape",
                "weekly_wape",
                "priority_count",
                "sales_coverage",
            ]
        ].to_string(index=False)
    )

    return best


# ============================================================
# FIGURE 1: FIVE-STORE SUMMARY
# ============================================================

def plot_five_store_summary(best):
    print("\nCreating five-store summary...")

    fig = plt.figure(
        figsize=(15, 10),
        facecolor="white",
    )

    ax = fig.add_axes(
        [0.08, 0.48, 0.89, 0.42]
    )

    positions = np.arange(
        len(best)
    )

    values = best[
        "weekly_wape"
    ].to_numpy()

    bars = ax.bar(
        positions,
        values,
        width=0.52,
        color=PREDICTED_COLOR,
    )

    ax.set_xticks(positions)

    ax.set_xticklabels(
        [
            f"Store {int(store_id)}"
            for store_id in best["store_id"]
        ],
        fontsize=12,
    )

    ax.set_ylabel(
        "Weekly product-total WAPE (%)",
        fontsize=12,
    )

    ax.set_ylim(
        0,
        max(values) * 1.35,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
    )

    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.035,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    fig.suptitle(
        "Five-Store Case Study | 7-Day Sales Forecast",
        fontsize=20,
        fontweight="bold",
        y=0.965,
    )

    # --------------------------------------------------------
    # METRICS TABLE
    # --------------------------------------------------------

    table_ax = fig.add_axes(
        [0.08, 0.18, 0.89, 0.24]
    )

    table_ax.axis("off")

    columns = [
        "Store",
        "Daily WAPE",
        "Weekly WAPE",
        "Priority products",
        "Sales coverage",
    ]

    table_rows = []

    for _, row in best.iterrows():

        priority_percentage = (
            row["priority_count"]
            / row["priority_total"]
            * 100.0
        )

        table_rows.append([
            str(int(row["store_id"])),
            f'{row["daily_wape"]:.2f}%',
            f'{row["weekly_wape"]:.2f}%',
            (
                f'{int(row["priority_count"])}/'
                f'{int(row["priority_total"])} '
                f'({priority_percentage:.0f}%)'
            ),
            f'{row["sales_coverage"]:.1f}%',
        ])

    table = table_ax.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[
            0.13,
            0.19,
            0.19,
            0.26,
            0.23,
        ],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.15)

    for (row, col), cell in table.get_celld().items():

        cell.set_edgecolor("#DDDDDD")

        if row == 0:
            cell.set_facecolor("#EAF3EF")
            cell.set_text_props(
                weight="bold"
            )

        elif row % 2 == 0:
            cell.set_facecolor("#F5F7F6")

        else:
            cell.set_facecolor("white")

    fig.text(
        0.5,
        0.065,
        (
            "Priority: historical weekly scaled sales >= 5. "
            "Sales coverage: share of each store's "
            "actual weekly sales represented by "
            "the selected 50 products."
        ),
        ha="center",
        fontsize=10,
        color="#666666",
    )

    output_path = (
        OUTPUT_DIR
        / "five_store_final_summary.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# FIGURE 2: TOP 30 PRODUCTS PER STORE
# ============================================================

def plot_store_top30(store_id, selected):
    print(
        f"Creating Top-30 chart for Store {store_id}..."
    )

    products = selected.loc[
        selected["store_id"] == store_id
    ].copy()

    products = (
        products.sort_values(
            ["actual_sale", "product_id"],
            ascending=[False, True],
        )
        .head(N_DISPLAY_PRODUCTS)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(12, 14),
        facecolor="white",
    )

    positions = np.arange(
        len(products)
    )

    bar_height = 0.36

    actual = products[
        "actual_sale"
    ].to_numpy()

    predicted = products[
        "predicted_sale"
    ].to_numpy()

    ax.barh(
        positions + bar_height / 2,
        actual,
        height=bar_height,
        color=ACTUAL_COLOR,
        label="Actual Sales",
    )

    ax.barh(
        positions - bar_height / 2,
        predicted,
        height=bar_height,
        color=PREDICTED_COLOR,
        label="Predicted Sales",
    )

    ax.set_yticks(positions)

    ax.set_yticklabels(
        [
            f"Product {int(product_id)}"
            for product_id in products["product_id"]
        ]
    )

    ax.invert_yaxis()

    maximum = max(
        float(actual.max()),
        float(predicted.max()),
    )

    ax.set_xlim(
        0,
        maximum * 1.2,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.15,
    )

    ax.set_axisbelow(True)

    for i, (actual_value, predicted_value) in enumerate(
        zip(actual, predicted)
    ):

        ax.text(
            actual_value + maximum * 0.008,
            i + bar_height / 2,
            f"{actual_value:.1f}",
            va="center",
            fontsize=8,
        )

        ax.text(
            predicted_value + maximum * 0.008,
            i - bar_height / 2,
            f"{predicted_value:.1f}",
            va="center",
            fontsize=8,
        )

    ax.set_title(
        f"Store {int(store_id)} | Top 30 Products\n"
        "Actual vs Predicted Weekly Sales",
        fontsize=17,
        fontweight="bold",
        pad=22,
    )

    ax.set_xlabel(
        "7-day total scaled sales",
        fontsize=11,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Top 30 products ranked by actual weekly sales. "
            "Period: 26 June - 2 July 2024. "
            "Sales are anonymized and scaled."
        ),
        ha="center",
        fontsize=9,
        color="#666666",
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    output_path = (
        OUTPUT_DIR
        / f"store_{int(store_id)}_top30_products.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged, history = load_data()

    selected = build_store_product_data(
        merged,
        history,
    )

    best = select_best_stores(
        selected,
        merged,
    )

    metrics_path = (
        OUTPUT_DIR / "selected_store_metrics.csv"
    )

    best.to_csv(
        metrics_path,
        index=False,
    )

    print(f"\nSaved: {metrics_path}")

    plot_five_store_summary(best)

    for store_id in best["store_id"]:
        plot_store_top30(
            store_id,
            selected,
        )

    print("\n=== VISUALIZATION COMPLETED ===")
    print("Five-store summary charts: 1")
    print("Store Top-30 charts: 5")
    print("Total PNG files: 6")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

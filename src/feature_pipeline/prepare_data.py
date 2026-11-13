from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.parquet"
PAIRS_PATH = DATA_DIR / "selected_pairs.csv"
OUTPUT_PATH = DATA_DIR / "train_selected.parquet"

N_PRODUCTS = 100
MIN_STORES_PER_PRODUCT = 50
EXPECTED_DAYS = 90


def select_products():
    print("=== SELECTING PRODUCTS ===")

    parquet = pq.ParquetFile(TRAIN_PATH)

    product_stores = defaultdict(set)
    product_categories = {}

    columns = [
        "store_id",
        "product_id",
        "first_category_id",
    ]

    for batch in parquet.iter_batches(
        batch_size=20_000,
        columns=columns,
    ):
        data = batch.to_pydict()

        for store, product, category in zip(
            data["store_id"],
            data["product_id"],
            data["first_category_id"],
        ):
            product_stores[product].add(store)

            if product in product_categories:
                if product_categories[product] != category:
                    raise ValueError(
                        f"Inconsistent category for product {product}"
                    )
            else:
                product_categories[product] = category

    eligible_products = [
        product
        for product, stores in product_stores.items()
        if len(stores) >= MIN_STORES_PER_PRODUCT
    ]

    print(f"Total products: {len(product_stores)}")
    print(
        f"Eligible products: {len(eligible_products)}"
    )

    if len(eligible_products) < N_PRODUCTS:
        raise ValueError(
            "Not enough eligible products."
        )

    category_products = defaultdict(list)

    for product in eligible_products:
        category = product_categories[product]
        category_products[category].append(product)

    # Prefer products with broad store coverage.
    for category in category_products:
        category_products[category].sort(
            key=lambda product: (
                -len(product_stores[product]),
                product,
            )
        )

    # Select products across categories in rounds.
    selected_products = []

    while len(selected_products) < N_PRODUCTS:
        added = False

        for category in sorted(category_products):
            if not category_products[category]:
                continue

            product = category_products[category].pop(0)
            selected_products.append(product)
            added = True

            if len(selected_products) == N_PRODUCTS:
                break

        if not added:
            break

    if len(selected_products) != N_PRODUCTS:
        raise ValueError(
            "Product selection did not reach 100 products."
        )

    selected_categories = Counter(
        product_categories[product]
        for product in selected_products
    )

    print(f"Selected products: {len(selected_products)}")
    print(
        f"Selected categories: {len(selected_categories)}"
    )

    return selected_products, product_stores


def expand_store_product_pairs(
    selected_products,
    product_stores,
):
    print("\n=== EXPANDING STORE-PRODUCT SERIES ===")

    pairs = []

    for product in selected_products:
        for store in sorted(product_stores[product]):
            pairs.append((store, product))

    expanded = pd.DataFrame(
        pairs,
        columns=["store_id", "product_id"],
    )

    expanded = (
        expanded.sort_values(
            ["product_id", "store_id"]
        )
        .reset_index(drop=True)
    )

    if expanded.duplicated(
        ["store_id", "product_id"]
    ).any():
        raise ValueError("Duplicate store-product pairs.")

    print(f"Products: {expanded['product_id'].nunique()}")
    print(f"Stores: {expanded['store_id'].nunique()}")
    print(f"Series: {len(expanded):,}")

    return expanded


def create_training_subset(expanded, selected_products):
    print("\n=== CREATING TRAINING SUBSET ===")

    dataset = ds.dataset(
        TRAIN_PATH,
        format="parquet",
    )

    table = dataset.to_table(
        filter=ds.field("product_id").isin(
            selected_products
        )
    )

    expected_rows = len(expanded) * EXPECTED_DAYS

    print(f"Expected rows: {expected_rows:,}")
    print(f"Actual rows: {table.num_rows:,}")

    if table.num_rows != expected_rows:
        raise ValueError(
            "Unexpected training row count."
        )

    actual_pairs = (
        table.select(
            ["store_id", "product_id"]
        )
        .to_pandas()
        .drop_duplicates()
        .sort_values(
            ["store_id", "product_id"]
        )
        .reset_index(drop=True)
    )

    expected_pairs = (
        expanded[
            ["store_id", "product_id"]
        ]
        .sort_values(
            ["store_id", "product_id"]
        )
        .reset_index(drop=True)
    )

    if not actual_pairs.equals(expected_pairs):
        raise ValueError(
            "Store-product pairs do not match."
        )

    print("\nSaving output files...")

    expanded.to_csv(
        PAIRS_PATH,
        index=False,
    )

    pq.write_table(
        table,
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n=== DATA PREPARATION COMPLETED ===")
    print(f"Products: {len(selected_products)}")
    print(f"Stores: {expanded['store_id'].nunique()}")
    print(f"Series: {len(expanded):,}")
    print(f"Training rows: {table.num_rows:,}")
    print(f"Training columns: {table.num_columns}")
    print(f"Days per series: {EXPECTED_DAYS}")

    print("\nSaved files:")
    print(PAIRS_PATH)
    print(OUTPUT_PATH)


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    selected_products, product_stores = select_products()

    expanded = expand_store_product_pairs(
        selected_products,
        product_stores,
    )

    create_training_subset(
        expanded,
        selected_products,
    )


if __name__ == "__main__":
    main()

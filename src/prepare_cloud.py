"""Prepare retail training data on Azure ML compute."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-train",
        required=True,
        help="Path to the raw training Parquet file.",
    )

    parser.add_argument(
        "--prepared-output",
        required=True,
        help="Directory for prepared output artifacts.",
    )

    args = parser.parse_args()

    source_dir = Path(__file__).resolve().parent / "feature_pipeline"

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Feature preparation code not found: {source_dir}"
        )

    raw_train = Path(args.raw_train)

    if not raw_train.is_file():
        raise FileNotFoundError(
            f"Raw training data not found: {raw_train}"
        )

    with tempfile.TemporaryDirectory() as temporary_dir:
        work_dir = Path(temporary_dir)

        shutil.copytree(
            source_dir,
            work_dir,
            dirs_exist_ok=True,
        )

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            raw_train,
            data_dir / "train.parquet",
        )

        subprocess.run(
            [sys.executable, str(work_dir / "prepare_data.py")],
            cwd=work_dir,
            check=True,
        )

        output_dir = Path(args.prepared_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = [
            "train_selected.parquet",
            "selected_pairs.csv",
        ]

        for filename in output_files:
            source = data_dir / filename

            if not source.is_file():
                raise FileNotFoundError(
                    f"Expected output not found: {source}"
                )

            shutil.copy2(
                source,
                output_dir / filename,
            )

        print(
            "Cloud data preparation completed successfully.",
            flush=True,
        )


if __name__ == "__main__":
    main()

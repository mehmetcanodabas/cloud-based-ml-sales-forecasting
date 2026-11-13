"""Run retail forecasting feature engineering on Azure ML."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepared", required=True)
    parser.add_argument("--raw-eval", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--forecast-output", required=True)

    args = parser.parse_args()

    source_dir = Path(__file__).resolve().parent / "feature_pipeline"

    if not (source_dir / "build_features.py").is_file():
        raise FileNotFoundError(
            f"Feature engineering script not found: {source_dir}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)

        shutil.copytree(
            source_dir,
            work_dir,
            dirs_exist_ok=True,
        )

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            Path(args.prepared) / "train_selected.parquet",
            data_dir / "train_selected.parquet",
        )

        shutil.copy2(
            Path(args.raw_eval),
            data_dir / "eval.parquet",
        )

        subprocess.run(
            [sys.executable, str(work_dir / "build_features.py"), "train"],
            cwd=work_dir,
            check=True,
        )

        subprocess.run(
            [sys.executable, str(work_dir / "build_features.py"), "forecast"],
            cwd=work_dir,
            check=True,
        )

        training_output = Path(args.training_output)
        forecast_output = Path(args.forecast_output)

        training_output.mkdir(parents=True, exist_ok=True)
        forecast_output.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            data_dir / "training_features_candidate.parquet",
            training_output / "training_features.parquet",
        )

        shutil.copy2(
            data_dir / "final_forecast_features.parquet",
            forecast_output / "final_forecast_features.parquet",
        )

    print(
        "Azure ML feature engineering completed successfully.",
        flush=True,
    )


if __name__ == "__main__":
    main()

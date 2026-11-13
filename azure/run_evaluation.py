"""
Cloud-Based ML Sales Forecasting
Azure ML - Final Evaluation

Runs evaluation on the held-out seven-day period.

Uses completed cloud predictions and actual sales.
Does not retrain the model or rerun inference.
"""

from pathlib import Path

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SUBSCRIPTION_ID = (
    "860cc196-1e4b-424c-81dd-4b96313997fc"
)

RESOURCE_GROUP = "rg-cloud-based-ml-sales-forecasting"
WORKSPACE_NAME = "mlw-sales-forecasting"
COMPUTE_NAME = "cpu-cluster"

ENVIRONMENT = "azureml:cloud-sales-forecast-env:1"

RAW_EVAL = ROOT / "raw_data" / "eval.parquet"

PREDICTIONS_DIR = ROOT / "evidence" / "inference"


# ============================================================
# FIND INPUT FILES
# ============================================================

def find_prediction_file():

    files = list(
        PREDICTIONS_DIR.rglob(
            "final_predictions.csv"
        )
    )

    if len(files) != 1:
        raise FileNotFoundError(
            "Expected exactly one final_predictions.csv "
            f"inside {PREDICTIONS_DIR}"
        )

    predictions_file = files[0]

    if predictions_file.stat().st_size == 0:
        raise ValueError(
            "Predictions file is empty."
        )

    return predictions_file


# ============================================================
# CONNECT TO AZURE
# ============================================================

def connect():

    client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    workspace = client.workspaces.get(
        WORKSPACE_NAME
    )

    compute = client.compute.get(
        COMPUTE_NAME
    )

    if (
        compute.min_instances != 0
        or compute.max_instances != 1
    ):
        raise RuntimeError(
            "Unexpected CPU scaling configuration."
        )

    print(f"Workspace: {workspace.name}")
    print(f"Compute: {compute.name}")

    return client


# ============================================================
# EVALUATION JOB
# ============================================================

def main():

    print("\n=== AZURE ML FINAL EVALUATION ===")

    predictions_file = find_prediction_file()

    if not RAW_EVAL.is_file():
        raise FileNotFoundError(
            f"Evaluation data not found: {RAW_EVAL}"
        )

    print(f"Predictions: {predictions_file}")
    print(f"Actual sales: {RAW_EVAL}")

    client = connect()

    evaluation_job = command(
        display_name="Final Sales Forecast Evaluation",
        experiment_name="cloud-sales-final-evaluation",
        code=str(ROOT / "src"),
        environment=ENVIRONMENT,
        compute=COMPUTE_NAME,
        command=(
            "python evaluate.py "
            "--predictions ${{inputs.predictions}} "
            "--actual-data ${{inputs.actual_data}} "
            "--output ${{outputs.metrics}}"
        ),
        inputs={
            "predictions": Input(
                type=AssetTypes.URI_FILE,
                path=str(predictions_file),
                mode="download",
            ),
            "actual_data": Input(
                type=AssetTypes.URI_FILE,
                path=str(RAW_EVAL),
                mode="download",
            ),
        },
        outputs={
            "metrics": Output(
                type=AssetTypes.URI_FOLDER,
                mode="rw_mount",
            ),
        },
    )

    print("\nSubmitting evaluation job...")

    submitted = client.jobs.create_or_update(
        evaluation_job
    )

    print("\n=== JOB SUBMITTED ===")
    print(f"Job ID: {submitted.name}")

    if submitted.studio_url:
        print(f"Azure Studio: {submitted.studio_url}")

    print(
        "\nCheck the job status in Azure ML Studio "
        "or with Azure CLI."
    )

    print(
        "Training and batch inference were not rerun."
    )


if __name__ == "__main__":
    main()

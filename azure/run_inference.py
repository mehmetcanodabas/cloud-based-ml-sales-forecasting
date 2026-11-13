"""
Cloud-Based ML Sales Forecasting
Azure ML - Batch Inference

Uses the registered LightGBM model and
forecast features already stored in Azure.

Does NOT retrain the model.
"""

from pathlib import Path

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential


# ============================================================
# AZURE CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SUBSCRIPTION_ID = (
    "860cc196-1e4b-424c-81dd-4b96313997fc"
)

RESOURCE_GROUP = "rg-cloud-based-ml-sales-forecasting"
WORKSPACE_NAME = "mlw-sales-forecasting"
COMPUTE_NAME = "cpu-cluster"

ENVIRONMENT = "azureml:cloud-sales-forecast-env:1"
MODEL_NAME = "cloud-retail-lightgbm"
MODEL_VERSION = "1"

FORECAST_DATA_URI = (
    "azureml://datastores/workspaceblobstore/paths/"
    "azureml/7d7bd317-e74f-44f2-9442-77cb3f7e5a91/"
    "forecast_data/"
)


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

    workspace = client.workspaces.get(WORKSPACE_NAME)
    compute = client.compute.get(COMPUTE_NAME)

    print(f"Workspace: {workspace.name}")
    print(f"Compute: {compute.name}")

    if (
        compute.min_instances != 0
        or compute.max_instances != 1
    ):
        raise RuntimeError(
            "Unexpected CPU cluster scaling configuration."
        )

    return client


# ============================================================
# VERIFY REGISTERED MODEL
# ============================================================

def verify_model(client):

    model = client.models.get(
        name=MODEL_NAME,
        version=MODEL_VERSION,
    )

    print(
        f"Registered model: {model.name}:{model.version}"
    )

    return (
        f"azureml:{model.name}:{model.version}"
    )


# ============================================================
# SUBMIT BATCH INFERENCE
# ============================================================

def main():

    print("\n=== AZURE ML BATCH INFERENCE ===")

    client = connect()

    model_reference = verify_model(client)

    # Verify that the workspace datastore exists.
    datastore = client.datastores.get(
        "workspaceblobstore"
    )

    print(f"Datastore: {datastore.name}")
    print(f"Forecast input: {FORECAST_DATA_URI}")

    inference_job = command(
        display_name="Cloud Batch Inference",
        experiment_name="cloud-sales-batch-inference",
        code=str(ROOT / "src"),
        environment=ENVIRONMENT,
        compute=COMPUTE_NAME,
        command=(
            "python batch_infer.py "
            "--model-dir ${{inputs.model}} "
            "--forecast-data ${{inputs.forecast_data}} "
            "--predictions-output ${{outputs.predictions}}"
        ),
        inputs={
            "model": Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=model_reference,
                mode="download",
            ),
            "forecast_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=FORECAST_DATA_URI,
                mode="ro_mount",
            ),
        },
        outputs={
            "predictions": Output(
                type=AssetTypes.URI_FOLDER,
                mode="rw_mount",
            ),
        },
    )

    print("\nSubmitting batch inference job...")

    submitted = client.jobs.create_or_update(
        inference_job
    )

    print(f"Job ID: {submitted.name}")

    if submitted.studio_url:
        print(f"Studio URL: {submitted.studio_url}")

    client.jobs.stream(submitted.name)

    completed = client.jobs.get(
        submitted.name
    )

    print(f"\nJob status: {completed.status}")

    if completed.status != "Completed":
        raise RuntimeError(
            "Batch inference was not completed successfully."
        )

    print("\n=== DOWNLOADING PREDICTIONS ===")

    download_dir = ROOT / "evidence" / "inference"

    download_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client.jobs.download(
        name=submitted.name,
        output_name="predictions",
        download_path=str(download_dir),
    )

    files = list(
        download_dir.rglob("final_predictions.csv")
    )

    if len(files) != 1:
        raise RuntimeError(
            "Expected one downloaded prediction CSV."
        )

    print(f"Predictions downloaded: {files[0]}")
    print("\nBATCH INFERENCE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

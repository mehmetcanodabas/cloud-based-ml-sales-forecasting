"""
Cloud-Based ML Sales Forecasting
Azure Machine Learning SDK v2

Workflow:
1. Data preparation
2. Feature engineering
3. LightGBM training
4. Model registration
5. Batch inference
6. Final evaluation

All ML processing runs on Azure CPU compute.
No online endpoint is created.
"""

import argparse
import json
from pathlib import Path

from azure.ai.ml import MLClient, Input, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment, Model
from azure.identity import AzureCliCredential


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SUBSCRIPTION_ID = "860cc196-1e4b-424c-81dd-4b96313997fc"

RESOURCE_GROUP = "rg-cloud-based-ml-sales-forecasting"
WORKSPACE_NAME = "mlw-sales-forecasting"
COMPUTE_NAME = "cpu-cluster"

ENVIRONMENT_NAME = "cloud-sales-forecast-env"
MODEL_NAME = "cloud-retail-lightgbm"

EXPERIMENT_NAME = "cloud-sales-forecasting"

CODE_DIRECTORY = ROOT / "src"
CONDA_FILE = ROOT / "configs" / "conda.yml"

RAW_TRAIN = ROOT / "raw_data" / "train.parquet"
RAW_EVAL = ROOT / "raw_data" / "eval.parquet"

EVIDENCE_DIR = ROOT / "evidence"


# ============================================================
# LOCAL FILE VALIDATION
# ============================================================

def check_project_files():
    print("\n=== CHECKING PROJECT FILES ===")

    required_files = [
        CONDA_FILE,
        RAW_TRAIN,
        RAW_EVAL,
        CODE_DIRECTORY / "prepare_cloud.py",
        CODE_DIRECTORY / "features_cloud.py",
        CODE_DIRECTORY / "train.py",
        CODE_DIRECTORY / "batch_infer.py",
        CODE_DIRECTORY / "evaluate.py",
        CODE_DIRECTORY / "feature_pipeline" / "prepare_data.py",
        CODE_DIRECTORY / "feature_pipeline" / "build_features.py",
    ]

    missing = []

    for file_path in required_files:
        if file_path.is_file():
            print(f"OK: {file_path.relative_to(ROOT)}")
        else:
            missing.append(file_path)

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(str(p) for p in missing)
        )

    for path in [RAW_TRAIN, RAW_EVAL]:
        if path.stat().st_size == 0:
            raise ValueError(
                f"Input file is empty: {path}"
            )

    print("\nAll required files exist.")


# ============================================================
# AZURE CONNECTION
# ============================================================

def connect_to_azure():
    print("\n=== CONNECTING TO AZURE ML ===")

    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    workspace = ml_client.workspaces.get(
        WORKSPACE_NAME
    )

    print(f"Workspace: {workspace.name}")
    print(f"Region: {workspace.location}")

    if workspace.location.lower() != "germanywestcentral":
        raise RuntimeError(
            "Unexpected Azure ML workspace region."
        )

    return ml_client


# ============================================================
# COMPUTE VALIDATION
# ============================================================

def check_compute(ml_client):
    print("\n=== CHECKING CPU COMPUTE ===")

    compute = ml_client.compute.get(
        COMPUTE_NAME
    )

    print(f"Compute: {compute.name}")
    print(f"VM size: {compute.size}")
    print(f"Minimum nodes: {compute.min_instances}")
    print(f"Maximum nodes: {compute.max_instances}")

    if (
        compute.min_instances != 0
        or compute.max_instances != 1
    ):
        raise RuntimeError(
            "CPU cluster must have "
            "min_instances=0 and max_instances=1."
        )

    print("Compute autoscaling configuration verified.")

    return compute


# ============================================================
# AZURE ML ENVIRONMENT
# ============================================================

def create_environment(ml_client):
    print("\n=== REGISTERING AZURE ML ENVIRONMENT ===")

    environment = Environment(
        name=ENVIRONMENT_NAME,
        description=(
            "Python 3.11 environment for "
            "LightGBM retail sales forecasting"
        ),
        image=(
            "mcr.microsoft.com/azureml/"
            "openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file=str(CONDA_FILE),
    )

    registered = (
        ml_client.environments.create_or_update(
            environment
        )
    )

    reference = (
        f"azureml:{registered.name}:"
        f"{registered.version}"
    )

    print(f"Environment: {reference}")

    return reference


# ============================================================
# DATA PREPARATION COMPONENT
# ============================================================

def create_preparation_component(environment):

    return command(
        name="prepare_retail_data",
        display_name="01 Data Preparation",
        code=str(CODE_DIRECTORY),
        environment=environment,
        command=(
            "python prepare_cloud.py "
            "--raw-train ${{inputs.raw_train}} "
            "--prepared-output ${{outputs.prepared}}"
        ),
        inputs={
            "raw_train": Input(
                type=AssetTypes.URI_FILE,
            ),
        },
        outputs={
            "prepared": Output(
                type=AssetTypes.URI_FOLDER,
            ),
        },
    )


# ============================================================
# FEATURE ENGINEERING COMPONENT
# ============================================================

def create_feature_component(environment):

    return command(
        name="build_retail_features",
        display_name="02 Feature Engineering",
        code=str(CODE_DIRECTORY),
        environment=environment,
        command=(
            "python features_cloud.py "
            "--prepared ${{inputs.prepared}} "
            "--raw-eval ${{inputs.raw_eval}} "
            "--training-output ${{outputs.training_data}} "
            "--forecast-output ${{outputs.forecast_data}}"
        ),
        inputs={
            "prepared": Input(
                type=AssetTypes.URI_FOLDER,
            ),
            "raw_eval": Input(
                type=AssetTypes.URI_FILE,
            ),
        },
        outputs={
            "training_data": Output(
                type=AssetTypes.URI_FOLDER,
            ),
            "forecast_data": Output(
                type=AssetTypes.URI_FOLDER,
            ),
        },
    )


# ============================================================
# TRAINING COMPONENT
# ============================================================

def create_training_component(environment):

    return command(
        name="train_retail_lightgbm",
        display_name="03 LightGBM Training",
        code=str(CODE_DIRECTORY),
        environment=environment,
        command=(
            "python train.py "
            "--training-data ${{inputs.training_data}} "
            "--model-output ${{outputs.model_artifacts}}"
        ),
        inputs={
            "training_data": Input(
                type=AssetTypes.URI_FOLDER,
            ),
        },
        outputs={
            "model_artifacts": Output(
                type=AssetTypes.URI_FOLDER,
            ),
        },
    )


# ============================================================
# WAIT FOR JOB
# ============================================================

def wait_for_job(ml_client, job_name):
    print(f"\nWaiting for Azure job: {job_name}")

    ml_client.jobs.stream(job_name)

    completed_job = ml_client.jobs.get(
        job_name
    )

    status = completed_job.status

    print(f"Job status: {status}")

    if status != "Completed":
        raise RuntimeError(
            f"Azure job failed or was not completed: "
            f"{job_name} ({status})"
        )

    return completed_job


# ============================================================
# TRAINING PIPELINE
# ============================================================

def run_training_pipeline(
    ml_client,
    environment,
):
    print("\n=== BUILDING TRAINING PIPELINE ===")

    preparation_component = (
        create_preparation_component(environment)
    )

    feature_component = (
        create_feature_component(environment)
    )

    training_component = (
        create_training_component(environment)
    )

    @dsl.pipeline(
        name="cloud-sales-forecasting",
        description=(
            "Data preparation, feature engineering "
            "and LightGBM training"
        ),
        default_compute=COMPUTE_NAME,
    )
    def forecasting_pipeline(raw_train, raw_eval):

        preparation = preparation_component(
            raw_train=raw_train,
        )

        features = feature_component(
            prepared=preparation.outputs.prepared,
            raw_eval=raw_eval,
        )

        training = training_component(
            training_data=(
                features.outputs.training_data
            ),
        )

        return {
            "trained_model": (
                training.outputs.model_artifacts
            ),
            "forecast_features": (
                features.outputs.forecast_data
            ),
        }

    pipeline_job = forecasting_pipeline(
        raw_train=Input(
            type=AssetTypes.URI_FILE,
            path=str(RAW_TRAIN),
        ),
        raw_eval=Input(
            type=AssetTypes.URI_FILE,
            path=str(RAW_EVAL),
        ),
    )

    pipeline_job.experiment_name = (
        EXPERIMENT_NAME
    )

    print("\nSubmitting training pipeline...")

    submitted = ml_client.jobs.create_or_update(
        pipeline_job
    )

    print(f"Pipeline job ID: {submitted.name}")

    if submitted.studio_url:
        print(f"Studio URL: {submitted.studio_url}")

    completed = wait_for_job(
        ml_client,
        submitted.name,
    )

    return completed


# ============================================================
# MODEL REGISTRATION
# ============================================================

def register_model(
    ml_client,
    training_job_name,
):
    print("\n=== REGISTERING LIGHTGBM MODEL ===")

    model_output_uri = (
        f"azureml://jobs/{training_job_name}"
        "/outputs/trained_model"
    )

    model = Model(
        name=MODEL_NAME,
        type=AssetTypes.CUSTOM_MODEL,
        path=model_output_uri,
        description=(
            "LightGBM model for seven-day "
            "retail sales forecasting"
        ),
        tags={
            "training_pipeline_job": (
                training_job_name
            ),
            "forecast_horizon": "7",
            "training_cutoff": "2024-06-25",
            "algorithm": "LightGBM",
        },
    )

    registered = (
        ml_client.models.create_or_update(
            model
        )
    )

    model_reference = (
        f"azureml:{registered.name}:"
        f"{registered.version}"
    )

    # Verify that the registered version is accessible.
    verified = ml_client.models.get(
        name=registered.name,
        version=registered.version,
    )

    print(f"Registered model: {model_reference}")
    print(f"Verified version: {verified.version}")

    return registered, model_reference


# ============================================================
# BATCH INFERENCE
# ============================================================

def run_batch_inference(
    ml_client,
    environment,
    model_reference,
    training_job_name,
):
    print("\n=== CLOUD BATCH INFERENCE ===")

    forecast_uri = (
        f"azureml://jobs/{training_job_name}"
        "/outputs/forecast_features"
    )

    inference_job = command(
        display_name="04 Batch Inference",
        experiment_name=(
            "cloud-sales-batch-inference"
        ),
        code=str(CODE_DIRECTORY),
        environment=environment,
        compute=COMPUTE_NAME,
        command=(
            "python batch_infer.py "
            "--model-dir ${{inputs.registered_model}} "
            "--forecast-data ${{inputs.forecast_features}} "
            "--predictions-output ${{outputs.predictions}}"
        ),
        inputs={
            "registered_model": Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=model_reference,
                mode="download",
            ),
            "forecast_features": Input(
                type=AssetTypes.URI_FOLDER,
                path=forecast_uri,
            ),
        },
        outputs={
            "predictions": Output(
                type=AssetTypes.URI_FOLDER,
            ),
        },
    )

    submitted = ml_client.jobs.create_or_update(
        inference_job
    )

    print(f"Inference job ID: {submitted.name}")

    return wait_for_job(
        ml_client,
        submitted.name,
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

def run_evaluation(
    ml_client,
    environment,
    inference_job_name,
):
    print("\n=== FINAL EVALUATION ===")

    predictions_uri = (
        f"azureml://jobs/{inference_job_name}"
        "/outputs/predictions"
    )

    evaluation_job = command(
        display_name="05 Final Evaluation",
        experiment_name=(
            "cloud-sales-final-evaluation"
        ),
        code=str(CODE_DIRECTORY),
        environment=environment,
        compute=COMPUTE_NAME,
        command=(
            "python evaluate.py "
            "--predictions ${{inputs.predictions}} "
            "--actual-data ${{inputs.actual_data}} "
            "--output ${{outputs.metrics}}"
        ),
        inputs={
            "predictions": Input(
                type=AssetTypes.URI_FOLDER,
                path=predictions_uri,
            ),
            "actual_data": Input(
                type=AssetTypes.URI_FILE,
                path=str(RAW_EVAL),
            ),
        },
        outputs={
            "metrics": Output(
                type=AssetTypes.URI_FOLDER,
            ),
        },
    )

    submitted = ml_client.jobs.create_or_update(
        evaluation_job
    )

    print(f"Evaluation job ID: {submitted.name}")

    return wait_for_job(
        ml_client,
        submitted.name,
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

def download_evidence(
    ml_client,
    inference_job_name,
    evaluation_job_name,
):
    print("\n=== DOWNLOADING RESULTS ===")

    prediction_dir = (
        EVIDENCE_DIR / "predictions"
    )

    metrics_dir = (
        EVIDENCE_DIR / "metrics"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ml_client.jobs.download(
        name=inference_job_name,
        output_name="predictions",
        download_path=str(prediction_dir),
    )

    ml_client.jobs.download(
        name=evaluation_job_name,
        output_name="metrics",
        download_path=str(metrics_dir),
    )

    prediction_files = list(
        prediction_dir.rglob(
            "final_predictions.csv"
        )
    )

    metric_files = list(
        metrics_dir.rglob(
            "evaluation_metrics.json"
        )
    )

    if len(prediction_files) != 1:
        raise RuntimeError(
            "Expected exactly one predictions CSV."
        )

    if len(metric_files) != 1:
        raise RuntimeError(
            "Expected exactly one evaluation metrics JSON."
        )

    metrics = json.loads(
        metric_files[0].read_text(
            encoding="utf-8"
        )
    )

    print(f"Predictions: {prediction_files[0]}")
    print(f"Metrics: {metric_files[0]}")

    return metrics


# ============================================================
# SAVE WORKFLOW METADATA
# ============================================================

def save_manifest(
    training_job,
    registered_model,
    inference_job,
    evaluation_job,
    metrics,
):
    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "workspace": WORKSPACE_NAME,
        "resource_group": RESOURCE_GROUP,
        "compute": COMPUTE_NAME,
        "training_pipeline": {
            "job_id": training_job.name,
            "status": training_job.status,
            "studio_url": training_job.studio_url,
        },
        "registered_model": {
            "name": registered_model.name,
            "version": registered_model.version,
        },
        "batch_inference": {
            "job_id": inference_job.name,
            "status": inference_job.status,
            "studio_url": inference_job.studio_url,
        },
        "evaluation": {
            "job_id": evaluation_job.name,
            "status": evaluation_job.status,
            "studio_url": evaluation_job.studio_url,
        },
        "metrics": metrics,
    }

    path = EVIDENCE_DIR / "run_manifest.json"

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Workflow manifest saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Azure ML retail sales "
            "forecasting workflow."
        )
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate project files and Azure "
            "resources without submitting jobs."
        ),
    )

    args = parser.parse_args()

    check_project_files()

    ml_client = connect_to_azure()

    check_compute(ml_client)

    if args.check_only:
        print(
            "\nPreflight checks passed. "
            "No Azure jobs were submitted."
        )
        return

    # Register environment.
    environment = create_environment(
        ml_client
    )

    # Data preparation -> Features -> Training.
    training_job = run_training_pipeline(
        ml_client,
        environment,
    )

    # Register final model.
    registered_model, model_reference = register_model(
        ml_client,
        training_job.name,
    )

    # Generate seven-day forecasts.
    inference_job = run_batch_inference(
        ml_client,
        environment,
        model_reference,
        training_job.name,
    )

    # Evaluate the held-out final week.
    evaluation_job = run_evaluation(
        ml_client,
        environment,
        inference_job.name,
    )

    # Download evidence.
    metrics = download_evidence(
        ml_client,
        inference_job.name,
        evaluation_job.name,
    )

    save_manifest(
        training_job=training_job,
        registered_model=registered_model,
        inference_job=inference_job,
        evaluation_job=evaluation_job,
        metrics=metrics,
    )

    print("\n====================================")
    print("AZURE ML WORKFLOW COMPLETED")
    print("====================================")

    print(
        f"Model: {registered_model.name}:"
        f"{registered_model.version}"
    )

    print(f"Training job: {training_job.name}")
    print(f"Inference job: {inference_job.name}")
    print(f"Evaluation job: {evaluation_job.name}")

    print(f"Evidence: {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()

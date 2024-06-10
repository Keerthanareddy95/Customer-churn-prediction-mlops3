import numpy as np
import pandas as pd
import json
from .utils import get_data_for_test
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""
    return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters"""
    pipeline_name: str
    step_name: str
    running: bool = True
    model_name: str = "model"

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline."""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "account_length",
        "international_plan",
        "voice_mail_plan",
        "number_vmail_messages",
        "total_day_calls",
        "total_eve_calls",
        "total_night_calls",
        "total_intl_calls",
        "number_customer_service_calls",
        "area_code_encoded",
        "total_day_minutes",
        "total_eve_minutes",
        "total_night_minutes",
        "total_intl_minutes",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction

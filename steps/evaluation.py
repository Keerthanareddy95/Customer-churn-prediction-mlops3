import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from src.evaluation import MSE, RMSE, R2

import mlflow
from zenml.client import Client
from typing import Tuple
from typing_extensions import Annotated

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model( model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    
    try:
        # prediction = model.predict(x_test)
        # evaluation = Evaluation()
        # r2_score = evaluation.r2_score(y_test, prediction)
        # mlflow.log_metric("r2_score", r2_score)
        # mse = evaluation.mean_squared_error(y_test, prediction)
        # mlflow.log_metric("mse", mse)
        # rmse = np.sqrt(mse)
        # mlflow.log_metric("rmse", rmse)

        prediction = model.predict(X_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2)
        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        
        return r2, rmse
    except Exception as e:
        logging.error("Error in evaluating the model: {}".format(e))
        raise e
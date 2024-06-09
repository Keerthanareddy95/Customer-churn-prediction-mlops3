import logging
from zenml import step
import pandas as pd

@step
def evaluate_model(df:pd.DataFrame) -> None:
    pass
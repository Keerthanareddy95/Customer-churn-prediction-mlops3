import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
class DataPreprocessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            # Remove the next line to keep the "churn" column
            # data = data.select_dtypes(include=[np.number])
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["total_day_charge", "total_eve_charge","total_night_charge","total_intl_charge"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(e)
            raise e

class DataDivideStrategy:
    def handle_data(self, data):
        try:
            logging.info("Data columns before division: %s", data.columns)
            X = data.drop("churn", axis=1)
            y = data["churn"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in handling data: %s", e)
            raise e

class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """Initializes the DataCleaning class with a specific strategy."""
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
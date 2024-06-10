import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str):
        """Initialize the data ingestion class."""
        self.data_path = data_path  # Store data_path as an instance variable

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        data = pd.read_csv(self.data_path)
        print("Columns in the ingested dataframe: ", data.columns.tolist())
        print(data.head())  # Print the first few rows of the dataframe for verification
        return data

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Args:
        Data path
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e

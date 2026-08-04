import pandas as pd

class DatasetProfiler:
    """
    A class to profile a pandas DataFrame, providing various statistics and information about the dataset.
    """
    
    def __init__(self):
        """
        Initializes the DatasetProfiler class.
        """
        pass
    
    @classmethod
    def profile(cls, df: pd.DataFrame) -> dict:
        """
        Profiles a pandas DataFrame and returns a dictionary containing various statistics and information.

        Args:
            df (pd.DataFrame): The pandas DataFrame to be profiled.

        Returns:
            dict: A dictionary containing the following keys:
                  - 'number_of_rows': Number of rows in the DataFrame
                  - 'number_of_columns': Number of columns in the DataFrame
                  - 'column_names': List of column names
                  - 'data_types': Dictionary with column names as keys and data types as values
                  - 'numeric_columns': List of numeric column names
                  - 'categorical_columns': List of categorical column names
                  - 'datetime_columns': List of datetime column names
                  - 'missing_values_per_column': Dictionary with column names as keys and missing value counts as values
                  - 'duplicate_row_count': Count of duplicate rows
                  - 'memory_usage': Memory usage of the DataFrame in bytes
        """
        profile = {}
        
        # Basic statistics
        profile['number_of_rows'] = df.shape[0]
        profile['number_of_columns'] = df.shape[1]
        profile['column_names'] = df.columns.tolist()
        profile['data_types'] = dict(df.dtypes)
        
        # Categorize columns
        profile['numeric_columns'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        profile['categorical_columns'] = df.select_dtypes(include=['object']).columns.tolist()
        profile['datetime_columns'] = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Missing values
        profile['missing_values_per_column'] = dict(df.isna().sum())
        
        # Duplicate rows
        profile['duplicate_row_count'] = df.duplicated().sum()
        
        # Memory usage
        profile['memory_usage'] = df.memory_usage(deep=True).sum()
        
        return profile
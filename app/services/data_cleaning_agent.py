import pandas as pd

class DataCleaningAgent:
    """
    A class for cleaning data in a pandas DataFrame.
    
    Methods:
        clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
            Cleans the input DataFrame by removing duplicates and empty rows,
            standardizing column names, and generating a cleaning summary.
    """

    def __init__(self):
        pass

    @staticmethod
    def clean_data(df: pd.DataFrame) -> tuple:
        """
        Cleans the input DataFrame by removing duplicates and empty rows,
        standardizing column names, and generating a cleaning summary.

        Parameters:
            df (pd.DataFrame): The input DataFrame to be cleaned.

        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]:
                A tuple containing the cleaned DataFrame and the cleaning summary as a dictionary.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Remove duplicate rows
        original_rows = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = original_rows - df.shape[0]

        # Remove completely empty rows
        original_rows = df.shape[0]
        df = df.dropna(how='all')
        empty_rows_removed = original_rows - df.shape[0]

        # Standardize column names (lowercase, replace spaces with underscores)
        original_columns = len(df.columns)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        columns_renamed = original_columns - len(df.columns)

        # Generate cleaning summary
        cleaning_summary = {
            "duplicates_removed": duplicates_removed,
            "empty_rows_removed": empty_rows_removed,
            "columns_renamed": columns_renamed,
        }

        return df, cleaning_summary

# Example usage:
if __name__ == "__main__":
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
        'Age': [25, 30, None, 25],
        'City': ['New York', 'Los Angeles', 'Chicago', 'New York']
    }
    df = pd.DataFrame(data)

    agent = DataCleaningAgent()
    cleaned_df, summary = agent.clean_data(df)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaning Summary:")
    print(summary)

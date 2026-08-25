import pandas as pd
from app.services.data_cleaning_agent import DataCleaningAgent

# Sample dataset
data = {
    "Name": ["Alice", "Bob", "Bob", None],
    "Age": [25, 30, 30, None],
    "Department": ["HR", "IT", "IT", None]
}

df = pd.DataFrame(data)

# Clean the dataset
cleaned_df, summary = DataCleaningAgent.clean_data(df)

# Print cleaned DataFrame
print("===== Cleaned DataFrame =====")
print(cleaned_df)

print("\n===== Cleaning Summary =====")
for key, value in summary.items():
    print(f"{key}: {value}")
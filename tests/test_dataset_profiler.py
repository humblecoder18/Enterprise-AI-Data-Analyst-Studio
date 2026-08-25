import pandas as pd
from app.services.dataset_profiler import DatasetProfiler

# Create a sample dataset
data = {
    "Name": ["Alice", "Bob", "Charlie", "Alice"],
    "Age": [25, 30, 35, 25],
    "Salary": [50000, 60000, None, 50000],
    "Department": ["HR", "IT", "Finance", "HR"]
}

df = pd.DataFrame(data)

# Profile the dataset
result = DatasetProfiler.profile(df)

# Print the results
for key, value in result.items():
    print(f"{key}: {value}")
from app.services.dataset_profiler import DatasetProfiler
from app.services.eda_agent import EDAAgent
from app.services.ml_recommendation_agent import MLRecommendationAgent

import pandas as pd

df = pd.DataFrame({
    "Age": [25, 30, 35, 40],
    "Salary": [30000, 45000, 60000, 75000],
    "Department": ["HR", "Sales", "IT", "Finance"]
})

profile = DatasetProfiler.profile(df)
eda_results = EDAAgent(df).analyze()

agent = MLRecommendationAgent()

recommendations = agent.recommend(profile, eda_results)

print(recommendations)
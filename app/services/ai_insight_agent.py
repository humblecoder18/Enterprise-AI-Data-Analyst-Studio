import json
from app.services.ai_service import AIService


class AIInsightAgent:
    """
    Generates AI-powered business insights from EDA results using Groq.
    """

    def __init__(self):
        self.ai = AIService()

    def generate_insights(self, eda_results: dict) -> str:
        """
        Generate AI business insights using Groq.
        Optimized to stay within the Groq free-tier token limit.
        """
        summary = {
            "Shape": eda_results.get("shape"),
            "Missing Values": eda_results.get("missing_values"),
            "Numeric Columns": eda_results.get("numeric_columns"),
            "Categorical Columns": eda_results.get("categorical_columns"),
            "Correlation": eda_results.get("correlation"),
            "Outliers": eda_results.get("outliers")
        }

        prompt = f"""
You are a Senior Data Scientist and Business Intelligence Consultant.

Analyze the dataset summary below.

Dataset Summary:
{json.dumps(summary, indent=2, default=str)}

Generate a professional report using the following sections:
1. Executive Summary
2. Dataset Quality
3. Key Insights
4. Correlation Analysis
5. Missing Value Analysis
6. Potential Data Quality Issues
7. Suitable Machine Learning Models
8. Business Recommendations

Rules:
- Maximum 350 words.
- Use bullet points.
- Do not repeat values unnecessarily.
- Do not invent information.
- Base the answer only on the provided summary.
"""
        return self.ai.ask(prompt, max_tokens=450)
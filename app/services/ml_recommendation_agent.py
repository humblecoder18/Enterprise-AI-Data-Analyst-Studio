import json
from app.services.ai_service import AIService


class MLRecommendationAgent:
    """
    Generates ML recommendations using Groq.
    Optimized for the Groq free tier.
    """

    def __init__(self):
        self.ai = AIService()

    def recommend(self, profile: dict, eda_results: dict) -> str:
        summary = {
            "Rows": profile.get("number_of_rows"),
            "Columns": profile.get("number_of_columns"),
            "Numeric Columns": profile.get("numeric_columns"),
            "Categorical Columns": profile.get("categorical_columns"),
            "Missing Values": eda_results.get("missing_values"),
            "Correlation": eda_results.get("correlation")
        }

        prompt = f"""
You are a Senior Machine Learning Engineer.

Analyze this dataset summary.

{json.dumps(summary, indent=2, default=str)}

Generate a professional report using the following sections:
1. Problem Type
2. Possible Target Column
3. Data Preprocessing
4. Recommended ML Algorithms
5. Evaluation Metrics
6. Important Features
7. Risks
8. Next Steps

Rules:
- Maximum 400 words.
- Use bullet points.
- Do not invent information.
- Use only the supplied dataset summary.
- Keep the answer concise.
"""
        return self.ai.ask(prompt, max_tokens=450)
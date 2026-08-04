from app.services.ai_service import AIService


class PredictionAIExplainer:
    """
    Converts technical machine-learning prediction results
    and local feature contributions into a concise,
    human-readable business explanation using the shared
    Groq service.
    """

    def __init__(self):
        self.ai = AIService()

    # =========================================================
    # FORMAT FACTORS
    # =========================================================

    @staticmethod
    def _format_factors(factors, limit=3):
        """
        Convert explanation factors into compact text
        suitable for the LLM prompt.
        """

        if not factors:
            return "None identified."

        lines = []

        for factor in factors[:limit]:

            feature = factor.get(
                "Feature",
                "Unknown"
            )

            contribution = factor.get(
                "Contribution",
                0
            )

            feature_value = factor.get(
                "Feature Value",
                None
            )

            try:
                contribution = float(
                    contribution
                )

                contribution_text = (
                    f"{contribution:.4f}"
                )

            except Exception:
                contribution_text = str(
                    contribution
                )

            if feature_value is not None:

                lines.append(
                    f"- {feature}: "
                    f"value={feature_value}, "
                    f"contribution={contribution_text}"
                )

            else:

                lines.append(
                    f"- {feature}: "
                    f"contribution={contribution_text}"
                )

        return "\n".join(lines)

    # =========================================================
    # GENERATE BUSINESS EXPLANATION
    # =========================================================

    def explain(
        self,
        target_column,
        prediction,
        confidence=None,
        supporting_factors=None,
        opposing_factors=None,
        model_name=None
    ):
        """
        Generate a business-friendly explanation of a
        machine-learning prediction.
        """

        supporting_factors = (
            supporting_factors or []
        )

        opposing_factors = (
            opposing_factors or []
        )

        # -----------------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------------

        if confidence is not None:

            try:

                confidence_text = (
                    f"{float(confidence) * 100:.2f}%"
                )

            except Exception:

                confidence_text = str(
                    confidence
                )

        else:

            confidence_text = (
                "Not available"
            )

        # -----------------------------------------------------
        # FORMAT FACTORS
        # -----------------------------------------------------

        supporting_text = (
            self._format_factors(
                supporting_factors
            )
        )

        opposing_text = (
            self._format_factors(
                opposing_factors
            )
        )

        # -----------------------------------------------------
        # PROMPT
        # -----------------------------------------------------

        prompt = f"""
You are a Senior Data Scientist explaining a machine-learning
prediction to a non-technical business user.

Prediction Information:

Target:
{target_column}

Predicted Value:
{prediction}

Prediction Confidence:
{confidence_text}

Model:
{model_name if model_name else "Machine Learning Model"}

Factors Supporting the Predicted Class:
{supporting_text}

Factors Opposing the Predicted Class:
{opposing_text}


Your task:

Write a concise, professional business explanation of why the
model produced this prediction.

Rules:

1. Use ONLY the prediction information and factors provided above.

2. Do NOT invent employee information, business context, causes,
   correlations, or facts that are not provided.

3. Clearly distinguish between factors supporting the prediction
   and factors opposing the prediction.

4. Do NOT claim that a feature caused the outcome.

5. Use phrases such as:
   - "the model indicates"
   - "the model places weight on"
   - "this factor supports the prediction"
   - "this factor pushes against the prediction"

6. Do NOT describe a positive numerical contribution as
   necessarily good for the business. Positive means only that
   it supports the predicted class.

7. Do NOT describe a negative numerical contribution as
   necessarily bad. Negative means only that it pushes against
   the predicted class.

8. If the target or prediction meaning is unclear, do not assume
   what the class means.

9. Keep the explanation between approximately 100 and 180 words.

10. Use this structure:

### Prediction Summary
Briefly explain the predicted result and confidence.

### Key Supporting Factors
Explain the strongest factors supporting the predicted class.

### Factors Pushing Against the Prediction
Explain the strongest opposing factors.

### Business Interpretation
Give a short, cautious interpretation suitable for a
non-technical decision-maker.

Do not include mathematical formulas.
"""

        # -----------------------------------------------------
        # GROQ
        # -----------------------------------------------------

        return self.ai.ask(
            prompt,
            max_tokens=300
        )
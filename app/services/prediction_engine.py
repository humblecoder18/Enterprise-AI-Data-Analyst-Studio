import pandas as pd
import numpy as np




class PredictionEngine:
    """
    Makes predictions using the best trained AutoML pipeline.

    Supports:
    - Classification
    - Regression
    - Classification probabilities
    - Confidence score
    """

    def __init__(
        self,
        model,
        problem_type: str,
        target_column: str
    ):
        self.model = model
        self.problem_type = problem_type
        self.target_column = target_column

        if self.model is None:
            raise ValueError(
                "A trained model is required for prediction."
            )

        if self.problem_type not in [
            "classification",
            "regression"
        ]:
            raise ValueError(
                "Problem type must be either "
                "'classification' or 'regression'."
            )

    # =========================================================
    # PREPARE INPUT
    # =========================================================

    @staticmethod
    def prepare_input(input_data):

        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        if isinstance(input_data, pd.Series):
            return input_data.to_frame().T

        if isinstance(input_data, dict):
            return pd.DataFrame([input_data])

        raise ValueError(
            "Input data must be a dictionary, "
            "Pandas Series, or Pandas DataFrame."
        )

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(self, input_data):

        input_df = self.prepare_input(
            input_data
        )

        prediction = self.model.predict(
            input_df
        )

        predicted_value = prediction[0]

        # Convert NumPy values to normal Python values
        if isinstance(predicted_value, np.generic):
            predicted_value = predicted_value.item()

        result = {
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "prediction": predicted_value,
        }

        # =====================================================
        # CLASSIFICATION
        # =====================================================

        if self.problem_type == "classification":

            probabilities = None
            class_probabilities = None
            confidence = None

            # -------------------------------------------------
            # PREDICT PROBABILITIES
            # -------------------------------------------------

            if hasattr(
                self.model,
                "predict_proba"
            ):

                try:

                    probabilities = (
                        self.model.predict_proba(
                            input_df
                        )[0]
                    )

                    classes = (
                        self.model.classes_
                    )

                    class_probabilities = {}

                    for class_name, probability in zip(
                        classes,
                        probabilities
                    ):

                        if isinstance(
                            class_name,
                            np.generic
                        ):
                            class_name = (
                                class_name.item()
                            )

                        class_probabilities[
                            str(class_name)
                        ] = round(
                            float(probability),
                            4
                        )

                    confidence = round(
                        float(
                            np.max(probabilities)
                        ),
                        4
                    )

                except Exception:

                    class_probabilities = None
                    confidence = None

            result[
                "class_probabilities"
            ] = class_probabilities

            result[
                "confidence"
            ] = confidence

        # =====================================================
        # REGRESSION
        # =====================================================

        else:

            try:
                result["prediction"] = round(
                    float(predicted_value),
                    4
                )
            except Exception:
                pass

        return result

    # =========================================================
    # BATCH PREDICTION
    # =========================================================

    def predict_batch(
        self,
        input_df: pd.DataFrame
    ):

        if not isinstance(
            input_df,
            pd.DataFrame
        ):
            raise ValueError(
                "Batch prediction requires "
                "a Pandas DataFrame."
            )

        predictions = self.model.predict(
            input_df
        )

        output_df = input_df.copy()

        output_df[
            f"predicted_{self.target_column}"
        ] = predictions

        # -----------------------------------------------------
        # ADD CONFIDENCE FOR CLASSIFICATION
        # -----------------------------------------------------

        if (
            self.problem_type
            == "classification"
            and hasattr(
                self.model,
                "predict_proba"
            )
        ):

            try:

                probabilities = (
                    self.model.predict_proba(
                        input_df
                    )
                )

                output_df[
                    "prediction_confidence"
                ] = np.max(
                    probabilities,
                    axis=1
                )

            except Exception:
                pass

        return output_df
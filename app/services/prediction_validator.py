import pandas as pd


class PredictionValidator:
    """
    Validates user inputs before prediction.
    """

    @staticmethod
    def validate_input(input_data: dict, feature_schema: dict):

        errors = []
        warnings = []

        for feature, metadata in feature_schema.items():

            if feature not in input_data:

                errors.append(
                    f"Missing input: {feature}"
                )
                continue

            value = input_data[feature]

            # ==============================
            # Numeric Validation
            # ==============================

            if metadata["type"] == "numeric":

                try:
                    value = float(value)
                except Exception:
                    errors.append(
                        f"{feature} must be numeric."
                    )
                    continue

                if value < metadata["min"]:
                    warnings.append(
                        f"{feature} is below the training minimum "
                        f"({metadata['min']})."
                    )

                if value > metadata["max"]:
                    warnings.append(
                        f"{feature} is above the training maximum "
                        f"({metadata['max']})."
                    )

            # ==============================
            # Categorical Validation
            # ==============================

            else:

                categories = metadata.get(
                    "categories",
                    []
                )

                if categories and str(value) not in categories:

                    errors.append(
                        f"{feature} must be one of: "
                        + ", ".join(categories)
                    )

        return {

            "valid": len(errors) == 0,

            "errors": errors,

            "warnings": warnings,
        }
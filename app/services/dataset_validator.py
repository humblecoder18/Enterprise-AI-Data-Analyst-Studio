import pandas as pd


class DatasetValidator:
    """
    Enterprise Dataset Validation Service.

    This validator is used before:
    - EDA
    - AutoML
    - Prediction
    - AI Insights
    - Chat with Data

    Returns validation results instead of
    immediately raising exceptions so the UI
    can display user-friendly messages.
    """

    @staticmethod
    def validate_dataset(df: pd.DataFrame):

        errors = []
        warnings = []

        # =====================================================
        # DATASET EXISTS
        # =====================================================

        if df is None:

            errors.append(
                "No dataset has been loaded."
            )

            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
            }

        # =====================================================
        # DATAFRAME TYPE
        # =====================================================

        if not isinstance(df, pd.DataFrame):

            errors.append(
                "Input is not a valid pandas DataFrame."
            )

            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
            }

        # =====================================================
        # EMPTY DATASET
        # =====================================================

        if df.empty:

            errors.append(
                "Dataset is empty."
            )

        # =====================================================
        # ROW COUNT
        # =====================================================

        if len(df) < 10:

            warnings.append(
                "Dataset contains fewer than 10 rows. "
                "Machine learning results may be unreliable."
            )

        # =====================================================
        # COLUMN COUNT
        # =====================================================

        if df.shape[1] == 0:

            errors.append(
                "Dataset contains no columns."
            )

        # =====================================================
        # DUPLICATE COLUMNS
        # =====================================================

        duplicate_columns = (
            df.columns[df.columns.duplicated()]
            .tolist()
        )

        if duplicate_columns:

            errors.append(
                "Duplicate column names detected: "
                + ", ".join(duplicate_columns)
            )

        # =====================================================
        # CONSTANT COLUMNS
        # =====================================================

        constant_columns = []

        for column in df.columns:

            if df[column].nunique(dropna=False) <= 1:

                constant_columns.append(column)

        if constant_columns:

            warnings.append(
                "Constant columns detected: "
                + ", ".join(constant_columns)
            )

        # =====================================================
        # MISSING VALUES
        # =====================================================

        missing_total = int(
            df.isna().sum().sum()
        )

        if missing_total > 0:

            warnings.append(
                f"Dataset contains {missing_total} missing values."
            )

        # =====================================================
        # RESULT
        # =====================================================

        return {

            "valid": len(errors) == 0,

            "errors": errors,

            "warnings": warnings,

            "rows": len(df),

            "columns": len(df.columns),
        }

    # =========================================================
    # TARGET VALIDATION
    # =========================================================

    @staticmethod
    def validate_target(df, target_column):

        errors = []

        if target_column not in df.columns:

            errors.append(
                f"Target column '{target_column}' does not exist."
            )

        else:

            target = df[target_column]

            if target.dropna().empty:

                errors.append(
                    "Target column contains only missing values."
                )

            if target.nunique(dropna=True) <= 1:

                errors.append(
                    "Target column must contain at least two unique values."
                )

        return {

            "valid": len(errors) == 0,

            "errors": errors,
        }
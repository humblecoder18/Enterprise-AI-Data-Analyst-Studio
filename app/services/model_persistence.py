import io
import json
import joblib
from datetime import datetime


class ModelPersistence:
    """
    Handles saving and loading trained AutoML models.

    Stores:
    - Trained sklearn pipeline
    - Model metadata
    - Feature schema
    - Feature column information
    """

    # =========================================================
    # CREATE MODEL PACKAGE
    # =========================================================

    @staticmethod
    def create_model_package(automl_results: dict):
        """
        Create a complete serializable model package
        from AutoML results.
        """

        if not automl_results:
            raise ValueError(
                "AutoML results are not available."
            )

        trained_model = automl_results.get(
            "trained_model"
        )

        if trained_model is None:
            raise ValueError(
                "No trained model is available."
            )

        package = {
            # Actual fitted sklearn Pipeline
            "trained_model": trained_model,

            # Model information
            "best_model": automl_results.get(
                "best_model"
            ),

            "problem_type": automl_results.get(
                "problem_type"
            ),

            "target_column": automl_results.get(
                "target_column"
            ),

            # Model selection
            "selection_metric": automl_results.get(
                "selection_metric"
            ),

            "selection_score": automl_results.get(
                "selection_score"
            ),

            # Feature information
            "feature_columns": automl_results.get(
                "feature_columns",
                []
            ),

            "numeric_columns": automl_results.get(
                "numeric_columns",
                []
            ),

            "categorical_columns": automl_results.get(
                "categorical_columns",
                []
            ),

            "feature_schema": automl_results.get(
                "feature_schema",
                {}
            ),

            # Package metadata
            "package_version": "1.0",

            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        return package

    # =========================================================
    # EXPORT MODEL TO BYTES
    # =========================================================

    @classmethod
    def export_model(cls, automl_results: dict) -> bytes:
        """
        Convert trained model package into downloadable
        joblib bytes.
        """

        package = cls.create_model_package(
            automl_results
        )

        buffer = io.BytesIO()

        joblib.dump(
            package,
            buffer
        )

        buffer.seek(0)

        return buffer.getvalue()

    # =========================================================
    # LOAD MODEL FROM BYTES
    # =========================================================

    @staticmethod
    def load_model(model_bytes: bytes):
        """
        Load a previously exported model package.
        """

        if not model_bytes:
            raise ValueError(
                "Model file is empty."
            )

        buffer = io.BytesIO(
            model_bytes
        )

        package = joblib.load(
            buffer
        )

        if not isinstance(package, dict):
            raise ValueError(
                "Invalid model package."
            )

        required_fields = [
            "trained_model",
            "best_model",
            "problem_type",
            "target_column",
            "feature_columns",
            "feature_schema",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in package
        ]

        if missing_fields:
            raise ValueError(
                "Invalid model package. Missing fields: "
                + ", ".join(missing_fields)
            )

        return package

    # =========================================================
    # GET METADATA
    # =========================================================

    @staticmethod
    def get_metadata(package: dict):
        """
        Return model metadata without returning the
        actual sklearn model.
        """

        if not package:
            return {}

        return {
            "best_model": package.get(
                "best_model"
            ),

            "problem_type": package.get(
                "problem_type"
            ),

            "target_column": package.get(
                "target_column"
            ),

            "selection_metric": package.get(
                "selection_metric"
            ),

            "selection_score": package.get(
                "selection_score"
            ),

            "feature_count": len(
                package.get(
                    "feature_columns",
                    []
                )
            ),

            "created_at": package.get(
                "created_at"
            ),

            "package_version": package.get(
                "package_version"
            ),
        }

    # =========================================================
    # EXPORT METADATA JSON
    # =========================================================

    @classmethod
    def export_metadata_json(
        cls,
        automl_results: dict
    ) -> str:
        """
        Export human-readable model metadata as JSON.
        """

        package = cls.create_model_package(
            automl_results
        )

        metadata = cls.get_metadata(
            package
        )

        metadata[
            "feature_columns"
        ] = package.get(
            "feature_columns",
            []
        )

        metadata[
            "numeric_columns"
        ] = package.get(
            "numeric_columns",
            []
        )

        metadata[
            "categorical_columns"
        ] = package.get(
            "categorical_columns",
            []
        )

        metadata[
            "feature_schema"
        ] = package.get(
            "feature_schema",
            {}
        )

        return json.dumps(
            metadata,
            indent=4,
            default=str
        )
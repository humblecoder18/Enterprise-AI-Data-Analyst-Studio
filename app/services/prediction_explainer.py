import numpy as np
import pandas as pd


class PredictionExplainer:
    """
    Explain predictions produced by trained AutoML models.

    Supports:

    Global/model-level explanation:
    - feature_importances_
    - coef_

    Local/per-prediction explanation:
    - Linear models
    - Logistic Regression
    - Other coefficient-based sklearn estimators

    Pipeline support:
    - Attempts to extract preprocessing
    - Retrieves transformed feature names
    - Calculates transformed input contributions
    """

    def __init__(self, model):
        self.model = model

    # =========================================================
    # GET FINAL ESTIMATOR
    # =========================================================

    def _get_estimator(self):
        """
        Return the final estimator from a sklearn Pipeline.
        """

        if hasattr(self.model, "steps"):

            if len(self.model.steps) > 0:
                return self.model.steps[-1][1]

        return self.model

    # =========================================================
    # GET PREPROCESSOR
    # =========================================================

    def _get_preprocessor(self):
        """
        Return the preprocessing portion of a sklearn Pipeline.
        """

        if not hasattr(self.model, "steps"):
            return None

        if len(self.model.steps) <= 1:
            return None

        try:
            return self.model[:-1]

        except Exception:
            return None

    # =========================================================
    # GET TRANSFORMED FEATURE NAMES
    # =========================================================

    def _get_feature_names(
        self,
        fallback_features
    ):
        """
        Try to retrieve feature names after preprocessing.
        """

        fallback_features = list(
            fallback_features
        )

        # -----------------------------------------------------
        # Not a pipeline
        # -----------------------------------------------------

        if not hasattr(
            self.model,
            "named_steps"
        ):
            return fallback_features

        # -----------------------------------------------------
        # Try complete preprocessing pipeline
        # -----------------------------------------------------

        preprocessor = self._get_preprocessor()

        if (
            preprocessor is not None
            and hasattr(
                preprocessor,
                "get_feature_names_out"
            )
        ):

            try:

                names = (
                    preprocessor
                    .get_feature_names_out()
                )

                return [
                    str(name)
                    for name in names
                ]

            except Exception:
                pass

        # -----------------------------------------------------
        # Search individual steps
        # -----------------------------------------------------

        for _, step in (
            self.model.named_steps.items()
        ):

            if hasattr(
                step,
                "get_feature_names_out"
            ):

                try:

                    names = (
                        step
                        .get_feature_names_out()
                    )

                    return [
                        str(name)
                        for name in names
                    ]

                except Exception:
                    pass

        return fallback_features

    # =========================================================
    # CLEAN FEATURE NAME
    # =========================================================

    @staticmethod
    def _clean_feature_name(name):
        """
        Remove common sklearn preprocessing prefixes.
        """

        name = str(name)

        if "__" in name:
            name = name.split(
                "__",
                1
            )[1]

        return name

    # =========================================================
    # PREPARE INPUT
    # =========================================================

    @staticmethod
    def _prepare_input(input_data):
        """
        Convert prediction input into a one-row DataFrame.
        """

        if isinstance(
            input_data,
            pd.DataFrame
        ):
            return input_data.copy()

        if isinstance(
            input_data,
            dict
        ):
            return pd.DataFrame(
                [input_data]
            )

        raise TypeError(
            "input_data must be a dictionary "
            "or pandas DataFrame."
        )

    # =========================================================
    # TRANSFORM INPUT
    # =========================================================

    def _transform_input(
        self,
        input_data
    ):
        """
        Apply the trained model's preprocessing steps.
        """

        input_df = self._prepare_input(
            input_data
        )

        preprocessor = self._get_preprocessor()

        # -----------------------------------------------------
        # No preprocessing pipeline
        # -----------------------------------------------------

        if preprocessor is None:

            try:

                return np.asarray(
                    input_df,
                    dtype=float
                )

            except Exception:

                return None

        # -----------------------------------------------------
        # Pipeline preprocessing
        # -----------------------------------------------------

        try:

            transformed = (
                preprocessor.transform(
                    input_df
                )
            )

            # Sparse matrix
            if hasattr(
                transformed,
                "toarray"
            ):

                transformed = (
                    transformed.toarray()
                )

            return np.asarray(
                transformed,
                dtype=float
            )

        except Exception:

            return None

    # =========================================================
    # GLOBAL EXPLANATION
    # =========================================================

    def explain(
        self,
        input_data,
        feature_columns,
        top_n=10
    ):
        """
        Return model-level feature importance.

        This method is kept compatible with the existing
        prediction_tab.py implementation.
        """

        estimator = (
            self._get_estimator()
        )

        feature_names = (
            self._get_feature_names(
                feature_columns
            )
        )

        importances = None
        explanation_type = None

        # =====================================================
        # TREE-BASED MODELS
        # =====================================================

        if hasattr(
            estimator,
            "feature_importances_"
        ):

            importances = np.asarray(
                estimator.feature_importances_,
                dtype=float
            )

            explanation_type = (
                "Feature Importance"
            )

        # =====================================================
        # COEFFICIENT-BASED MODELS
        # =====================================================

        elif hasattr(
            estimator,
            "coef_"
        ):

            coefficients = np.asarray(
                estimator.coef_,
                dtype=float
            )

            # -------------------------------------------------
            # Binary / single-output
            # -------------------------------------------------

            if coefficients.ndim == 2:

                if coefficients.shape[0] == 1:

                    importances = np.abs(
                        coefficients[0]
                    )

                else:

                    # Multiclass
                    importances = np.mean(
                        np.abs(
                            coefficients
                        ),
                        axis=0
                    )

            else:

                importances = np.abs(
                    coefficients
                )

            explanation_type = (
                "Coefficient Importance"
            )

        # =====================================================
        # UNSUPPORTED MODEL
        # =====================================================

        else:

            return {
                "available": False,
                "message": (
                    "Global explainability is not "
                    "available for this model yet."
                ),
                "explanation_type": None,
                "top_features": []
            }

        importances = np.ravel(
            importances
        )

        # =====================================================
        # ALIGN FEATURE NAMES
        # =====================================================

        if (
            len(feature_names)
            != len(importances)
        ):

            feature_names = [
                f"Feature {i + 1}"
                for i in range(
                    len(importances)
                )
            ]

        # =====================================================
        # BUILD DATAFRAME
        # =====================================================

        explanation_df = pd.DataFrame(
            {
                "Feature": [
                    self._clean_feature_name(
                        feature
                    )
                    for feature
                    in feature_names
                ],

                "Importance":
                    importances
            }
        )

        explanation_df = (
            explanation_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(top_n)
            .reset_index(
                drop=True
            )
        )

        # =====================================================
        # NORMALIZE
        # =====================================================

        maximum = (
            explanation_df[
                "Importance"
            ].max()
            if not explanation_df.empty
            else 0
        )

        if maximum > 0:

            explanation_df[
                "Relative Importance (%)"
            ] = (
                explanation_df[
                    "Importance"
                ]
                / maximum
                * 100
            )

        else:

            explanation_df[
                "Relative Importance (%)"
            ] = 0.0

        return {
            "available": True,
            "explanation_type":
                explanation_type,
            "top_features":
                explanation_df.to_dict(
                    orient="records"
                )
        }

    # =========================================================
    # LOCAL EXPLANATION
    # =========================================================

    def explain_prediction(
        self,
        input_data,
        feature_columns,
        predicted_class=None,
        top_n=10
    ):
        """
        Explain one specific prediction.

        Currently provides true local contribution values for
        coefficient-based models such as Logistic Regression.

        Contribution:

            transformed_feature_value * model_coefficient

        Positive contribution:
            pushes the model score upward for the explained class.

        Negative contribution:
            pushes the model score downward.
        """

        estimator = (
            self._get_estimator()
        )

        # =====================================================
        # CHECK MODEL SUPPORT
        # =====================================================

        if not hasattr(
            estimator,
            "coef_"
        ):

            return {
                "available": False,
                "message": (
                    "Local prediction explanation is "
                    "currently available for coefficient-based "
                    "models such as Logistic Regression. "
                    "Tree-model local explanation will be "
                    "added separately."
                ),
                "explanation_type": None,
                "predicted_class":
                    predicted_class,
                "top_contributions": [],
                "increasing_factors": [],
                "decreasing_factors": []
            }

        # =====================================================
        # TRANSFORM INPUT
        # =====================================================

        transformed_input = (
            self._transform_input(
                input_data
            )
        )

        if transformed_input is None:

            return {
                "available": False,
                "message": (
                    "The prediction input could not be "
                    "transformed using the trained "
                    "preprocessing pipeline."
                ),
                "explanation_type": None,
                "predicted_class":
                    predicted_class,
                "top_contributions": [],
                "increasing_factors": [],
                "decreasing_factors": []
            }

        if transformed_input.ndim == 1:

            transformed_input = (
                transformed_input.reshape(
                    1,
                    -1
                )
            )

        feature_values = (
            transformed_input[0]
        )

        # =====================================================
        # GET COEFFICIENTS
        # =====================================================

        coefficients = np.asarray(
            estimator.coef_,
            dtype=float
        )

        classes = getattr(
            estimator,
            "classes_",
            None
        )

        explained_class = (
            predicted_class
        )

        # =====================================================
        # BINARY CLASSIFICATION
        # =====================================================

        if (
            coefficients.ndim == 2
            and coefficients.shape[0] == 1
        ):

            coefficient_vector = (
                coefficients[0]
            )

            # sklearn binary classifiers store coefficients
            # for classes_[1].
            if (
                classes is not None
                and len(classes) == 2
            ):

                positive_class = (
                    classes[1]
                )

                if predicted_class is None:

                    explained_class = (
                        positive_class
                    )

                elif str(
                    predicted_class
                ) != str(
                    positive_class
                ):

                    # Reverse direction when explaining
                    # the negative class.
                    coefficient_vector = (
                        -coefficient_vector
                    )

        # =====================================================
        # MULTICLASS CLASSIFICATION
        # =====================================================

        elif coefficients.ndim == 2:

            class_index = 0

            if (
                classes is not None
                and predicted_class is not None
            ):

                matching_indexes = [
                    index
                    for index, class_value
                    in enumerate(classes)
                    if str(class_value)
                    == str(predicted_class)
                ]

                if matching_indexes:

                    class_index = (
                        matching_indexes[0]
                    )

            coefficient_vector = (
                coefficients[
                    class_index
                ]
            )

            if classes is not None:

                try:

                    explained_class = (
                        classes[
                            class_index
                        ]
                    )

                except Exception:
                    pass

        # =====================================================
        # SINGLE-DIMENSION COEFFICIENT
        # =====================================================

        else:

            coefficient_vector = (
                np.ravel(
                    coefficients
                )
            )

        coefficient_vector = np.ravel(
            coefficient_vector
        )

        feature_values = np.ravel(
            feature_values
        )

        # =====================================================
        # VALIDATE DIMENSIONS
        # =====================================================

        if (
            len(coefficient_vector)
            != len(feature_values)
        ):

            return {
                "available": False,
                "message": (
                    "The transformed feature count does not "
                    "match the model coefficient count."
                ),
                "explanation_type": None,
                "predicted_class":
                    explained_class,
                "top_contributions": [],
                "increasing_factors": [],
                "decreasing_factors": []
            }

        # =====================================================
        # FEATURE NAMES
        # =====================================================

        feature_names = (
            self._get_feature_names(
                feature_columns
            )
        )

        if (
            len(feature_names)
            != len(feature_values)
        ):

            feature_names = [
                f"Feature {i + 1}"
                for i in range(
                    len(feature_values)
                )
            ]

        # =====================================================
        # LOCAL CONTRIBUTIONS
        # =====================================================

        contributions = (
            feature_values
            * coefficient_vector
        )

        local_df = pd.DataFrame(
            {
                "Feature": [
                    self._clean_feature_name(
                        feature
                    )
                    for feature
                    in feature_names
                ],

                "Feature Value":
                    feature_values,

                "Coefficient":
                    coefficient_vector,

                "Contribution":
                    contributions
            }
        )

        local_df[
            "Absolute Contribution"
        ] = np.abs(
            local_df[
                "Contribution"
            ]
        )

        # =====================================================
        # DIRECTION
        # =====================================================

        local_df[
            "Direction"
        ] = np.where(
            local_df[
                "Contribution"
            ] >= 0,
            "Increases prediction",
            "Decreases prediction"
        )

        # =====================================================
        # TOP CONTRIBUTIONS
        # =====================================================

        top_df = (
            local_df
            .sort_values(
                "Absolute Contribution",
                ascending=False
            )
            .head(top_n)
            .reset_index(
                drop=True
            )
        )

        # =====================================================
        # RELATIVE CONTRIBUTION
        # =====================================================

        maximum = (
            top_df[
                "Absolute Contribution"
            ].max()
            if not top_df.empty
            else 0
        )

        if maximum > 0:

            top_df[
                "Relative Contribution (%)"
            ] = (
                top_df[
                    "Absolute Contribution"
                ]
                / maximum
                * 100
            )

        else:

            top_df[
                "Relative Contribution (%)"
            ] = 0.0

        # =====================================================
        # INCREASING FACTORS
        # =====================================================

        increasing_df = (
            top_df[
                top_df[
                    "Contribution"
                ] > 0
            ]
            .sort_values(
                "Contribution",
                ascending=False
            )
        )

        # =====================================================
        # DECREASING FACTORS
        # =====================================================

        decreasing_df = (
            top_df[
                top_df[
                    "Contribution"
                ] < 0
            ]
            .sort_values(
                "Contribution",
                ascending=True
            )
        )

        # =====================================================
        # RETURN
        # =====================================================

        return {
            "available": True,

            "explanation_type":
                "Local Coefficient Contribution",

            "predicted_class":
                (
                    explained_class.item()
                    if isinstance(
                        explained_class,
                        np.generic
                    )
                    else explained_class
                ),

            "top_contributions":
                top_df.to_dict(
                    orient="records"
                ),

            "increasing_factors":
                increasing_df.to_dict(
                    orient="records"
                ),

            "decreasing_factors":
                decreasing_df.to_dict(
                    orient="records"
                )
        }
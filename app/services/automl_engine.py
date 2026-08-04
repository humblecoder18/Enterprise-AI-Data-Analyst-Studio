import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

# Classification Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Regression Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class AutoMLEngine:
    """
    Enterprise AutoML Engine.

    Features:
    - Automatic classification/regression detection
    - Missing-value handling
    - Numerical scaling
    - Categorical encoding
    - Automatic train/test split
    - Multiple model training
    - Model comparison
    - Best-model selection
    - Confusion matrix
    - Per-class classification metrics
    - Positive-class evaluation
    - ROC-AUC
    - Feature importance
    - Trained model preservation
    - Feature schema generation
    - Prediction engine support
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self, df: pd.DataFrame, target_column: str):

        self.df = df.copy()
        self.target_column = target_column

        self.problem_type = None
        self.preprocessor = None

        self.results = []
        self.trained_models = {}

        self.best_model = None
        self.best_model_name = None

        self.selection_metric = None
        self.selection_score = None

        self.test_data = {}
        self.classification_details = {}

        # SHAP support
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.feature_importance = None

        # Prediction-related information
        self.feature_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_schema = {}

    # =========================================================
    # DETECT PROBLEM TYPE
    # =========================================================

    def detect_problem_type(self):

        if self.target_column not in self.df.columns:

            raise ValueError(
                f"Target column '{self.target_column}' does not exist."
            )

        target = self.df[self.target_column]

        if (
            target.dtype == "object"
            or str(target.dtype) == "category"
            or str(target.dtype) == "bool"
        ):

            self.problem_type = "classification"

        elif target.nunique(dropna=True) <= 20:

            self.problem_type = "classification"

        else:

            self.problem_type = "regression"

        return self.problem_type

    # =========================================================
    # PREPARE DATA
    # =========================================================

    def prepare_data(self):

        if self.target_column not in self.df.columns:

            raise ValueError(
                f"Target column '{self.target_column}' does not exist."
            )

        # -----------------------------------------------------
        # Remove rows where target is missing
        # -----------------------------------------------------

        data = self.df.dropna(
            subset=[self.target_column]
        ).copy()

        if len(data) < 10:

            raise ValueError(
                "Dataset is too small for AutoML. "
                "At least 10 rows are required."
            )

        # -----------------------------------------------------
        # Split X and y
        # -----------------------------------------------------

        X = data.drop(
            columns=[self.target_column]
        )

        y = data[self.target_column]

        if X.shape[1] == 0:

            raise ValueError(
                "No feature columns are available."
            )

        # -----------------------------------------------------
        # Remove constant columns
        # -----------------------------------------------------

        constant_columns = [
            column
            for column in X.columns
            if X[column].nunique(dropna=False) <= 1
        ]

        if constant_columns:

            X = X.drop(
                columns=constant_columns
            )

        if X.shape[1] == 0:

            raise ValueError(
                "No usable feature columns remain "
                "after removing constant columns."
            )

        # =====================================================
        # SAVE FEATURE COLUMNS
        # =====================================================

        self.feature_columns = X.columns.tolist()

        # =====================================================
        # DETECT COLUMN TYPES
        # =====================================================

        self.numeric_columns = (
            X.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        self.categorical_columns = (
            X.select_dtypes(
                exclude=np.number
            )
            .columns
            .tolist()
        )

        # =====================================================
        # BUILD FEATURE SCHEMA
        # =====================================================

        self.feature_schema = self._build_feature_schema(X)

        # =====================================================
        # NUMERIC PIPELINE
        # =====================================================

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        # =====================================================
        # CATEGORICAL PIPELINE
        # =====================================================

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers = []

        if self.numeric_columns:

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns,
                )
            )

        if self.categorical_columns:

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns,
                )
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers
        )

        return X, y

    # =========================================================
    # BUILD FEATURE SCHEMA
    # =========================================================

    def _build_feature_schema(self, X: pd.DataFrame):
        """
        Build metadata about every feature.

        This schema will later be used by Streamlit to
        automatically create prediction input fields.
        """

        schema = {}

        for column in X.columns:

            series = X[column]

            # =================================================
            # NUMERIC FEATURE
            # =================================================

            if pd.api.types.is_numeric_dtype(series):

                clean_series = series.dropna()

                if clean_series.empty:

                    minimum = 0.0
                    maximum = 0.0
                    median = 0.0
                    mean = 0.0

                else:

                    minimum = float(
                        clean_series.min()
                    )

                    maximum = float(
                        clean_series.max()
                    )

                    median = float(
                        clean_series.median()
                    )

                    mean = float(
                        clean_series.mean()
                    )

                is_integer = (
                    pd.api.types.is_integer_dtype(
                        series
                    )
                )

                schema[column] = {

                    "type": "numeric",

                    "dtype": str(
                        series.dtype
                    ),

                    "is_integer": bool(
                        is_integer
                    ),

                    "min": minimum,

                    "max": maximum,

                    "median": median,

                    "mean": mean,
                }

            # =================================================
            # CATEGORICAL FEATURE
            # =================================================

            else:

                clean_series = (
                    series
                    .dropna()
                    .astype(str)
                )

                unique_values = (
                    clean_series
                    .unique()
                    .tolist()
                )

                # Avoid huge select boxes
                if len(unique_values) <= 100:

                    categories = sorted(
                        unique_values
                    )

                else:

                    categories = []

                mode = clean_series.mode()

                default_value = (
                    str(mode.iloc[0])
                    if not mode.empty
                    else ""
                )

                schema[column] = {

                    "type": "categorical",

                    "dtype": str(
                        series.dtype
                    ),

                    "categories": categories,

                    "default": default_value,

                    "unique_count": int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                }

        return schema

    # =========================================================
    # CLASSIFICATION MODELS
    # =========================================================

    @staticmethod
    def classification_models():

        return {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42,
                ),
        }

    # =========================================================
    # REGRESSION MODELS
    # =========================================================

    @staticmethod
    def regression_models():

        return {

            "Linear Regression":
                LinearRegression(),

            "Random Forest Regressor":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                ),

            "Gradient Boosting Regressor":
                GradientBoostingRegressor(
                    random_state=42,
                ),
        }

    # =========================================================
    # CLASSIFICATION TRAINING
    # =========================================================

    def train_classification(self, X, y):

        if y.nunique() < 2:

            raise ValueError(
                "Classification target must contain "
                "at least two classes."
            )

        value_counts = y.value_counts()

        stratify_target = (
            y
            if value_counts.min() >= 2
            else None
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify_target,
            )
        )

        self.test_data = {
            "X_test": X_test,
            "y_test": y_test,
        }

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        models = self.classification_models()

        for model_name, model in models.items():

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        self.preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(
                X_test
            )

            # =================================================
            # OVERALL METRICS
            # =================================================

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )

            # =================================================
            # LABELS
            # =================================================

            labels = list(
                pipeline.classes_
            )

            # =================================================
            # CONFUSION MATRIX
            # =================================================

            cm = confusion_matrix(
                y_test,
                predictions,
                labels=labels,
            )

            # =================================================
            # CLASSIFICATION REPORT
            # =================================================

            report = classification_report(
                y_test,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )

            # =================================================
            # ROC-AUC
            # =================================================

            roc_auc = None

            positive_class = None
            positive_precision = None
            positive_recall = None
            positive_f1 = None

            if len(labels) == 2:

                positive_class = labels[1]

                try:

                    probabilities = (
                        pipeline.predict_proba(
                            X_test
                        )[:, 1]
                    )

                    binary_y_test = (
                        y_test == positive_class
                    ).astype(int)

                    roc_auc = roc_auc_score(
                        binary_y_test,
                        probabilities,
                    )

                except Exception:

                    roc_auc = None

                # =============================================
                # POSITIVE CLASS METRICS
                # =============================================

                positive_class_key = str(
                    positive_class
                )

                if positive_class_key in report:

                    positive_precision = (
                        report[
                            positive_class_key
                        ]["precision"]
                    )

                    positive_recall = (
                        report[
                            positive_class_key
                        ]["recall"]
                    )

                    positive_f1 = (
                        report[
                            positive_class_key
                        ]["f1-score"]
                    )

            # =================================================
            # RESULT
            # =================================================

            result = {

                "Model": model_name,

                "Accuracy": round(
                    accuracy,
                    4
                ),

                "Precision": round(
                    precision,
                    4
                ),

                "Recall": round(
                    recall,
                    4
                ),

                "F1 Score": round(
                    f1,
                    4
                ),

                "ROC AUC": (
                    round(
                        roc_auc,
                        4
                    )
                    if roc_auc is not None
                    else None
                ),

                "Positive Precision": (
                    round(
                        positive_precision,
                        4
                    )
                    if positive_precision is not None
                    else None
                ),

                "Positive Recall": (
                    round(
                        positive_recall,
                        4
                    )
                    if positive_recall is not None
                    else None
                ),

                "Positive F1": (
                    round(
                        positive_f1,
                        4
                    )
                    if positive_f1 is not None
                    else None
                ),
            }

            self.results.append(
                result
            )

            # =================================================
            # SAVE TRAINED PIPELINE
            # =================================================

            self.trained_models[
                model_name
            ] = pipeline

            # =================================================
            # SAVE DETAILED EVALUATION
            # =================================================

            self.classification_details[
                model_name
            ] = {

                "labels": [
                    str(label)
                    for label in labels
                ],

                "confusion_matrix":
                    cm.tolist(),

                "classification_report":
                    report,

                "positive_class": (
                    str(
                        positive_class
                    )
                    if positive_class is not None
                    else None
                ),
            }

    # =========================================================
    # REGRESSION TRAINING
    # =========================================================

    def train_regression(self, X, y):

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )
        )

        self.test_data = {
            "X_test": X_test,
            "y_test": y_test,
        }

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        models = self.regression_models()

        for model_name, model in models.items():

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        self.preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(
                X_test
            )

            # =================================================
            # METRICS
            # =================================================

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            mse = mean_squared_error(
                y_test,
                predictions,
            )

            rmse = np.sqrt(
                mse
            )

            r2 = r2_score(
                y_test,
                predictions,
            )

            result = {

                "Model": model_name,

                "MAE": round(
                    mae,
                    4
                ),

                "RMSE": round(
                    rmse,
                    4
                ),

                "R2 Score": round(
                    r2,
                    4
                ),
            }

            self.results.append(
                result
            )

            # =================================================
            # SAVE TRAINED PIPELINE
            # =================================================

            self.trained_models[
                model_name
            ] = pipeline

    # =========================================================
    # SELECT BEST MODEL
    # =========================================================

    def select_best_model(self):

        if not self.results:

            raise ValueError(
                "No model results available."
            )

        results_df = pd.DataFrame(
            self.results
        )

        # =====================================================
        # CLASSIFICATION
        # =====================================================

        if self.problem_type == "classification":

            if (
                "ROC AUC" in results_df.columns
                and
                results_df[
                    "ROC AUC"
                ].notna().any()
            ):

                best_index = (
                    results_df[
                        "ROC AUC"
                    ]
                    .fillna(-1)
                    .idxmax()
                )

                self.selection_metric = (
                    "ROC AUC"
                )

                self.selection_score = (
                    results_df.loc[
                        best_index,
                        "ROC AUC"
                    ]
                )

            else:

                best_index = (
                    results_df[
                        "F1 Score"
                    ].idxmax()
                )

                self.selection_metric = (
                    "Weighted F1 Score"
                )

                self.selection_score = (
                    results_df.loc[
                        best_index,
                        "F1 Score"
                    ]
                )

        # =====================================================
        # REGRESSION
        # =====================================================

        else:

            best_index = (
                results_df[
                    "R2 Score"
                ].idxmax()
            )

            self.selection_metric = (
                "R2 Score"
            )

            self.selection_score = (
                results_df.loc[
                    best_index,
                    "R2 Score"
                ]
            )

        # =====================================================
        # SAVE BEST MODEL
        # =====================================================

        self.best_model_name = (
            results_df.loc[
                best_index,
                "Model"
            ]
        )

        self.best_model = (
            self.trained_models[
                self.best_model_name
            ]
        )

        return self.best_model_name

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================

    def calculate_feature_importance(self):

        if self.best_model is None:

            return None

        try:

            preprocessor = (
                self.best_model.named_steps[
                    "preprocessor"
                ]
            )

            model = (
                self.best_model.named_steps[
                    "model"
                ]
            )

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            importance_values = None

            # =================================================
            # TREE MODELS
            # =================================================

            if hasattr(
                model,
                "feature_importances_"
            ):

                importance_values = (
                    model.feature_importances_
                )

            # =================================================
            # LINEAR / LOGISTIC MODELS
            # =================================================

            elif hasattr(
                model,
                "coef_"
            ):

                coefficients = (
                    model.coef_
                )

                if coefficients.ndim == 1:

                    importance_values = (
                        np.abs(
                            coefficients
                        )
                    )

                else:

                    importance_values = (
                        np.mean(
                            np.abs(
                                coefficients
                            ),
                            axis=0,
                        )
                    )

            if importance_values is None:

                return None

            feature_importance_df = (
                pd.DataFrame(
                    {
                        "Feature":
                            feature_names,

                        "Importance":
                            importance_values,
                    }
                )
            )

            # =================================================
            # CLEAN FEATURE NAMES
            # =================================================

            feature_importance_df[
                "Feature"
            ] = (
                feature_importance_df[
                    "Feature"
                ]
                .str.replace(
                    "numeric__",
                    "",
                    regex=False,
                )
                .str.replace(
                    "categorical__",
                    "",
                    regex=False,
                )
            )

            feature_importance_df = (
                feature_importance_df
                .sort_values(
                    "Importance",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            self.feature_importance = (
                feature_importance_df
            )

            return feature_importance_df

        except Exception:

            return None

    # =========================================================
    # GET BEST TRAINED MODEL
    # =========================================================

    def get_best_model(self):

        if self.best_model is None:

            raise ValueError(
                "Best model is not available. "
                "Run AutoML training first."
            )

        return self.best_model

    # =========================================================
    # GET FEATURE SCHEMA
    # =========================================================

    def get_feature_schema(self):

        if not self.feature_schema:

            raise ValueError(
                "Feature schema is not available. "
                "Run AutoML training first."
            )

        return self.feature_schema

    # =========================================================
    # RUN COMPLETE AUTOML PIPELINE
    # =========================================================

    def run(self):

        # =====================================================
        # RESET STATE
        # =====================================================

        self.results = []
        self.trained_models = {}

        self.best_model = None
        self.best_model_name = None

        self.selection_metric = None
        self.selection_score = None

        self.classification_details = {}
        self.feature_importance = None

        self.feature_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_schema = {}

        # =====================================================
        # DETECT PROBLEM TYPE
        # =====================================================

        problem_type = (
            self.detect_problem_type()
        )

        # =====================================================
        # PREPARE DATA
        # =====================================================

        X, y = self.prepare_data()

        # =====================================================
        # TRAIN MODELS
        # =====================================================

        if problem_type == "classification":

            self.train_classification(
                X,
                y,
            )

        elif problem_type == "regression":

            self.train_regression(
                X,
                y,
            )

        else:

            raise ValueError(
                "Unable to determine "
                "machine learning problem type."
            )

        # =====================================================
        # SELECT BEST MODEL
        # =====================================================

        best_model_name = (
            self.select_best_model()
        )

        # =====================================================
        # FEATURE IMPORTANCE
        # =====================================================

        feature_importance = (
            self.calculate_feature_importance()
        )

        # =====================================================
        # RESULTS DATAFRAME
        # =====================================================

        results_df = pd.DataFrame(
            self.results
        )

        # =====================================================
        # OUTPUT
        # =====================================================

        output = {

            "problem_type":
                problem_type,

            "target_column":
                self.target_column,

            "best_model":
                best_model_name,

            "selection_metric":
                self.selection_metric,

            "selection_score":
                self.selection_score,

            "results":
                results_df,

            "feature_importance":
                feature_importance,

            # =================================================
            # PREDICTION SUPPORT
            # =================================================

            # Actual fitted sklearn pipeline
            "trained_model":
                self.best_model,

            # Features expected by model
            "feature_columns":
                self.feature_columns,

            # Numerical feature names
            "numeric_columns":
                self.numeric_columns,

            # Categorical feature names
            "categorical_columns":
                self.categorical_columns,

            # Metadata used to build prediction UI
            "feature_schema":
                self.feature_schema,

            "X_train":
                self.X_train,
            "X_test":
                self.X_test,
            "y_train":
                self.y_train,
            "y_test":
                self.y_test,
        }

        # =====================================================
        # CLASSIFICATION DETAILS
        # =====================================================

        if problem_type == "classification":

            output[
                "classification_details"
            ] = (
                self.classification_details
            )

            output[
                "best_model_details"
            ] = (
                self.classification_details.get(
                    best_model_name
                )
            )

        return output
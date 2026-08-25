import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.services.automl_engine import AutoMLEngine
from app.services.model_persistence import ModelPersistence
from app.services.dataset_validator import DatasetValidator



def render_automl_tab(cleaned_df: pd.DataFrame):
    """
    Render the AutoML interface.
    """

    # =========================================================
    # HEADER
    # =========================================================

    st.subheader("🧠 AutoML Model Training")

    st.write(
        "Select a target column and automatically train, compare, "
        "and evaluate multiple machine learning models."
    )

    if cleaned_df is None or cleaned_df.empty:
        st.warning("No dataset available for AutoML.")
        return

    st.divider()

    # =========================================================
    # TARGET COLUMN
    # =========================================================

    st.subheader("🎯 Select Target Column")

    st.caption("Choose the column that you want the model to predict.")

    target_column = st.selectbox(
        "Target Column",
        cleaned_df.columns.tolist(),
        key="automl_target_column"
    )

    target_series = cleaned_df[target_column]

    unique_values = target_series.nunique(dropna=True)

    # =========================================================
    # DETECT PROBLEM TYPE FOR UI
    # =========================================================

    if (
        target_series.dtype == "object"
        or str(target_series.dtype) == "category"
        or str(target_series.dtype) == "bool"
        or unique_values <= 20
    ):
        detected_problem = "Classification"
    else:
        detected_problem = "Regression"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎯 Target Column",
            target_column
        )

    with col2:
        st.metric(
            "🧠 Problem Type",
            detected_problem
        )

    with col3:
        st.metric(
            "🔢 Unique Values",
            unique_values
        )

    if detected_problem == "Classification":
        st.info(
            "Classification detected. "
            "The system will predict categories or classes."
        )
    else:
        st.info(
            "Regression detected. "
            "The system will predict continuous numerical values."
        )

    # =========================================================
    # TARGET DISTRIBUTION
    # =========================================================

    st.divider()

    st.subheader("📊 Target Distribution")

    try:

        if detected_problem == "Classification":

            counts = (
                target_series
                .astype(str)
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Target",
                "Count"
            ]

            fig = px.bar(
                counts,
                x="Target",
                y="Count",
                title=f"{target_column} Distribution"
            )

        else:

            fig = px.histogram(
                cleaned_df,
                x=target_column,
                title=f"{target_column} Distribution"
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"automl_target_distribution_{target_column}"
        )

    except Exception as e:
        st.warning(
            f"Unable to display target distribution: {e}"
        )

    # =========================================================
    # TRAINING
    # =========================================================

    st.divider()

    if st.button(
        "🚀 Train AutoML Models",
        key="train_automl_models",
        type="primary"
    ):

        try:

            with st.spinner(
                "Training and evaluating machine learning models..."
            ):

                # =====================================================
                # DATASET VALIDATION
                # =====================================================

                validation = DatasetValidator.validate_dataset(
                    cleaned_df
                )

                if not validation["valid"]:

                    for error in validation["errors"]:

                        st.error(error)

                    return

                for warning in validation["warnings"]:

                    st.warning(warning)

                # =====================================================
                # TARGET VALIDATION
                # =====================================================

                target_validation = (
                    DatasetValidator.validate_target(
                        cleaned_df,
                        target_column
                    )
                )

                if not target_validation["valid"]:

                    for error in target_validation["errors"]:

                        st.error(error)

                    return

                engine = AutoMLEngine(
                    cleaned_df,
                    target_column
                )

                automl_results = engine.run() 
                st.write("AutoML Keys")
                st.write(list(automl_results.keys()))
                st.write("trained_model")
                st.write(automl_results.get("trained_model"))
                st.write("X_train")
                st.write(automl_results.get("X_train"))
                
                # Store everything required for later prediction
                st.session_state["automl_results"] = automl_results

                st.session_state["automl_target"] = target_column

                st.session_state["automl_trained"] = True

            st.success(
                "✅ AutoML training completed successfully!"
            )

        except Exception as e:

            st.error(
                f"❌ AutoML training failed: {e}"
            )

            return

    # =========================================================
    # CHECK RESULTS
    # =========================================================

    if not st.session_state.get(
        "automl_trained",
        False
    ):
        st.info(
            "Select a target column and click "
            "'Train AutoML Models' to begin."
        )
        return

    automl_results = st.session_state.get(
        "automl_results"
    )

    if not automl_results:
        return

    # Prevent displaying old results for a different target
    if (
        st.session_state.get("automl_target")
        != target_column
    ):
        st.warning(
            "The selected target has changed. "
            "Click 'Train AutoML Models' to train models for this target."
        )
        return

    # =========================================================
    # RESULT VARIABLES
    # =========================================================

    problem_type = automl_results.get(
        "problem_type"
    )

    best_model = automl_results.get(
        "best_model"
    )

    selection_metric = automl_results.get(
        "selection_metric"
    )

    selection_score = automl_results.get(
        "selection_score"
    )

    results_df = automl_results.get(
        "results"
    )

    feature_importance = automl_results.get(
        "feature_importance"
    )

    # =========================================================
    # BEST MODEL
    # =========================================================

    st.divider()

    st.subheader("🏆 Best Model")

    best1, best2, best3 = st.columns(3)

    with best1:
        st.metric(
            "Best Model",
            best_model
        )

    with best2:
        st.metric(
            "Problem Type",
            str(problem_type).title()
        )

    with best3:

        if selection_score is not None:

            try:
                score_display = f"{float(selection_score):.4f}"
            except Exception:
                score_display = str(selection_score)

            st.metric(
                selection_metric or "Selection Score",
                score_display
            )

    if selection_metric:

        st.caption(
            f"The best model was selected using "
            f"**{selection_metric}**."
        )

    # =========================================================
    # MODEL COMPARISON TABLE
    # =========================================================

    st.divider()

    st.subheader("📊 Model Comparison")

    if (
        isinstance(results_df, pd.DataFrame)
        and not results_df.empty
    ):

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
        # AUTOML LEADERBOARD
        # =====================================================

        st.subheader("🏆 AutoML Leaderboard")

        leaderboard_df = results_df.copy()

        score_columns = [
            c for c in [
                "Accuracy",
                "F1 Score",
                "R2 Score"
            ] if c in leaderboard_df.columns
        ]

        if score_columns:
            sort_col = score_columns[0]
            leaderboard_df = leaderboard_df.sort_values(
                by=sort_col,
                ascending=False
            ).reset_index(drop=True)

            leaderboard_df.insert(
                0,
                "Rank",
                [f"#{i+1}" for i in range(len(leaderboard_df))]
            )

            st.dataframe(
                leaderboard_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Leaderboard CSV",
                leaderboard_df.to_csv(index=False),
                file_name="automl_leaderboard.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success(
                f"🥇 Best Model: {leaderboard_df.iloc[0]['Model']}"
            )


        # =====================================================
        # PERFORMANCE CHART
        # =====================================================

        st.subheader("📈 Model Performance")

        try:

            if problem_type == "classification":

                available_metrics = [
                    metric
                    for metric in [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "ROC AUC"
                    ]
                    if metric in results_df.columns
                ]

                if available_metrics:

                    performance_df = results_df[
                        ["Model"] + available_metrics
                    ].copy()

                    melted_df = performance_df.melt(
                        id_vars="Model",
                        var_name="Metric",
                        value_name="Score"
                    )

                    fig = px.bar(
                        melted_df,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Classification Model Performance"
                    )

                    fig.update_yaxes(
                        range=[0, 1]
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="automl_classification_performance"
                    )

            else:

                if "R2 Score" in results_df.columns:

                    fig = px.bar(
                        results_df,
                        x="Model",
                        y="R2 Score",
                        title="Regression Model R² Performance"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="automl_regression_r2"
                    )

                error_metrics = [
                    metric
                    for metric in [
                        "MAE",
                        "RMSE"
                    ]
                    if metric in results_df.columns
                ]

                if error_metrics:

                    error_df = results_df[
                        ["Model"] + error_metrics
                    ].melt(
                        id_vars="Model",
                        var_name="Metric",
                        value_name="Score"
                    )

                    error_fig = px.bar(
                        error_df,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Regression Error Comparison"
                    )

                    st.plotly_chart(
                        error_fig,
                        use_container_width=True,
                        key="automl_regression_errors"
                    )

        except Exception as e:

            st.warning(
                f"Unable to display performance chart: {e}"
            )

    # =========================================================
    # CLASSIFICATION DETAILS
    # =========================================================

    if problem_type == "classification":

        best_details = automl_results.get(
            "best_model_details"
        )

        # =====================================================
        # POSITIVE CLASS PERFORMANCE
        # =====================================================

        positive_columns = [
            column
            for column in [
                "Model",
                "Positive Precision",
                "Positive Recall",
                "Positive F1"
            ]
            if (
                isinstance(results_df, pd.DataFrame)
                and column in results_df.columns
            )
        ]

        if len(positive_columns) > 1:

            positive_df = results_df[
                positive_columns
            ].copy()

            if (
                positive_df
                .drop(columns=["Model"])
                .notna()
                .any()
                .any()
            ):

                st.divider()

                st.subheader(
                    "🎯 Positive-Class Performance"
                )

                st.caption(
                    "Useful for binary classification problems "
                    "where detecting the positive class is important."
                )

                st.dataframe(
                    positive_df,
                    use_container_width=True,
                    hide_index=True
                )

                positive_melted = (
                    positive_df.melt(
                        id_vars="Model",
                        var_name="Metric",
                        value_name="Score"
                    )
                    .dropna()
                )

                if not positive_melted.empty:

                    fig = px.bar(
                        positive_melted,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Positive-Class Model Comparison"
                    )

                    fig.update_yaxes(
                        range=[0, 1]
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="automl_positive_class_performance"
                    )

        # =====================================================
        # CONFUSION MATRIX
        # =====================================================

        if best_details:

            labels = best_details.get(
                "labels",
                []
            )

            matrix = best_details.get(
                "confusion_matrix"
            )

            if matrix is not None:

                st.divider()

                st.subheader(
                    "🧩 Confusion Matrix"
                )

                st.caption(
                    f"Confusion matrix for the selected best model: "
                    f"**{best_model}**"
                )

                try:

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=matrix,
                            x=labels,
                            y=labels,
                            text=matrix,
                            texttemplate="%{text}",
                            hovertemplate=(
                                "Predicted: %{x}<br>"
                                "Actual: %{y}<br>"
                                "Count: %{z}"
                                "<extra></extra>"
                            )
                        )
                    )

                    fig.update_layout(
                        title="Actual vs Predicted",
                        xaxis_title="Predicted Class",
                        yaxis_title="Actual Class"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="automl_confusion_matrix"
                    )

                except Exception as e:

                    st.warning(
                        f"Unable to display confusion matrix: {e}"
                    )

            # =================================================
            # CLASS LEVEL PERFORMANCE
            # =================================================

            report = best_details.get(
                "classification_report"
            )

            if report:

                rows = []

                for class_name, values in report.items():

                    if not isinstance(
                        values,
                        dict
                    ):
                        continue

                    if class_name in [
                        "accuracy",
                        "macro avg",
                        "weighted avg"
                    ]:
                        continue

                    rows.append(
                        {
                            "Class": class_name,
                            "Precision": values.get(
                                "precision"
                            ),
                            "Recall": values.get(
                                "recall"
                            ),
                            "F1 Score": values.get(
                                "f1-score"
                            ),
                            "Support": values.get(
                                "support"
                            )
                        }
                    )

                if rows:

                    class_df = pd.DataFrame(
                        rows
                    )

                    st.divider()

                    st.subheader(
                        "📋 Class-Level Performance"
                    )

                    st.dataframe(
                        class_df,
                        use_container_width=True,
                        hide_index=True
                    )

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================

    if (
        isinstance(
            feature_importance,
            pd.DataFrame
        )
        and not feature_importance.empty
    ):

        st.divider()

        st.subheader(
            "🔍 Feature Importance"
        )

        st.caption(
            "Features that contributed most strongly "
            "to the selected model."
        )

        top_features = (
            feature_importance
            .head(20)
            .sort_values(
                "Importance",
                ascending=True
            )
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 20 Important Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="automl_feature_importance"
        )

        with st.expander(
            "View Feature Importance Table"
        ):

            st.dataframe(
                feature_importance,
                use_container_width=True,
                hide_index=True
            )

    # =========================================================
    # PREDICTION READINESS
    # =========================================================

    st.divider()

    st.subheader(
        "🔮 Prediction System"
    )

    trained_model = automl_results.get(
        "trained_model"
    )

    feature_schema = automl_results.get(
        "feature_schema"
    )

    if (
        trained_model is not None
        and feature_schema
    ):

        st.success(
            "✅ Best model is trained and ready for predictions."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction Features",
                len(feature_schema)
            )

        with col2:
            st.metric(
                "Prediction Model",
                best_model
            )

        st.info(
            "The prediction engine is ready. "
            "The interactive prediction form will be added next."
        )

    else:

        st.warning(
            "Prediction model information is not available. "
            "Retrain AutoML models."
        )

    # =========================================================
    # AUTOML SUMMARY
    # =========================================================

    st.divider()

    st.subheader(
        "📋 AutoML Summary"
    )

    summary_col1, summary_col2 = (
        st.columns(2)
    )

    with summary_col1:

        st.write(
            f"**Target Column:** {target_column}"
        )

        st.write(
            f"**Problem Type:** "
            f"{str(problem_type).title()}"
        )

        st.write(
            f"**Best Model:** {best_model}"
        )

    with summary_col2:

        st.write(
            f"**Selection Metric:** "
            f"{selection_metric}"
        )

        if selection_score is not None:

            try:
                st.write(
                    f"**Selection Score:** "
                    f"{float(selection_score):.4f}"
                )
            except Exception:
                st.write(
                    f"**Selection Score:** "
                    f"{selection_score}"
                )

        st.write(
            f"**Models Compared:** "
            f"{len(results_df) if isinstance(results_df, pd.DataFrame) else 0}"
        )

    # =========================================================
    # MODEL EXPORT
    # =========================================================

    st.divider()

    st.subheader("💾 Export Trained Model")

    st.caption(
        "Download the trained AutoML model and its metadata "
        "for future use."
    )

    try:

        model_bytes = ModelPersistence.export_model(
            automl_results
        )

        metadata_json = (
            ModelPersistence.export_metadata_json(
                automl_results
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                label="⬇️ Download Model (.joblib)",
                data=model_bytes,
                file_name="enterprise_ai_model.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )

        with col2:

            st.download_button(
                label="⬇️ Download Metadata (.json)",
                data=metadata_json,
                file_name="enterprise_ai_model_metadata.json",
                mime="application/json",
                use_container_width=True,
            )

        st.success(
            "The trained model can now be reused without retraining."
        )

    except Exception as e:

        st.warning(
            f"Unable to export model: {e}"
        )

    # =========================================================
    # LOAD SAVED MODEL
    # =========================================================

    st.divider()

    st.subheader("📂 Load Saved Model")

    st.caption(
        "Upload a previously exported AutoML model package."
    )

    uploaded_model = st.file_uploader(
        "Choose a .joblib model file",
        type=["joblib"],
        key="uploaded_model_file"
    )

    if uploaded_model is not None:

        try:

            model_package = ModelPersistence.load_model(
                uploaded_model.read()
            )

            # Save the loaded package
            st.session_state["loaded_model_package"] = model_package

            # Make Prediction tab use this model
            st.session_state["automl_results"] = {
                "trained_model": model_package["trained_model"],
                "feature_schema": model_package["feature_schema"],
                "feature_columns": model_package["feature_columns"],
                "numeric_columns": model_package.get("numeric_columns", []),
                "categorical_columns": model_package.get("categorical_columns", []),
                "problem_type": model_package["problem_type"],
                "target_column": model_package["target_column"],
                "best_model": model_package["best_model"],
                "selection_metric": model_package.get("selection_metric"),
                "selection_score": model_package.get("selection_score"),
            }

            st.session_state["automl_trained"] = True

            st.success("✅ Model loaded successfully!")

            metadata = ModelPersistence.get_metadata(model_package)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Model", metadata["best_model"])
                st.metric("Problem Type", metadata["problem_type"])

            with col2:
                st.metric("Target", metadata["target_column"])
                st.metric("Features", metadata["feature_count"])

        except Exception as e:

            st.error(f"Unable to load model: {e}")
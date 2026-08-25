import streamlit as st
from app.services.shap_explainer import SHAPExplainer


def render_explainability_tab():

    st.title("🧩 Model Explainability")
    st.caption(
        "Understand how the trained machine learning model makes predictions using SHAP."
    )

    st.divider()

    automl_results = st.session_state.get("automl_results")

    if automl_results is None:
        st.warning("Please train an AutoML model first.")
        return

    model = automl_results.get("trained_model")
    X_train = automl_results.get("X_train")

    if model is None or X_train is None:
        st.warning("Training data is unavailable.")
        return

    with st.spinner("Initializing SHAP Explainer..."):
        explainer = SHAPExplainer(model, X_train)

    st.subheader("📊 Feature Importance")

    importance_df = explainer.feature_importance(X_train)

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        importance_df.set_index("Feature")["Importance"]
    )

    st.divider()

    st.subheader("📈 SHAP Summary Plot")

    try:
        st.pyplot(explainer.summary_plot(X_train))
    except Exception as e:
        st.warning(f"Unable to generate summary plot: {e}")

    st.divider()

    st.subheader("📊 SHAP Bar Plot")

    try:
        st.pyplot(explainer.bar_plot(X_train))
    except Exception as e:
        st.warning(f"Unable to generate SHAP bar plot: {e}")

    st.divider()

    st.subheader("🔍 Local Prediction Explanation")

    row_index = st.number_input(
        "Select Row",
        min_value=0,
        max_value=len(X_train)-1,
        value=0
    )

    try:
        table = explainer.shap_table(X_train, row_index)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(f"Unable to compute SHAP values: {e}")

    st.divider()

    st.subheader("🌊 Waterfall Plot")

    try:
        st.pyplot(explainer.waterfall_plot(X_train, row_index))
    except Exception as e:
        st.warning(f"Unable to generate waterfall plot: {e}")

    st.divider()

    st.subheader("📉 Dependence Plot")

    feature = st.selectbox(
        "Feature",
        explainer.display_feature_names,
        key="dependence_feature"
    )

    try:
        st.pyplot(explainer.dependence_plot(feature, X_train))
    except Exception as e:
        st.warning(f"Unable to generate dependence plot: {e}")

    st.divider()

    st.download_button(
        "⬇️ Download Feature Importance",
        importance_df[["Feature", "Importance"]].to_csv(index=False),
        file_name="feature_importance.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success("✅ Explainability analysis completed.")
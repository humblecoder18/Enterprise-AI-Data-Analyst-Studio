import streamlit as st

from app.services.file_loader import FileLoader
from app.services.dataset_profiler import DatasetProfiler
from app.services.data_cleaning_agent import DataCleaningAgent
from app.services.eda_agent import EDAAgent

from app.ui.sidebar import render_sidebar
from app.ui.dashboard_tab import render_dashboard_tab
from app.ui.analytics_tab import render_analytics_tab
from app.ui.ai_tab import render_ai_tab
from app.ui.dataset_chat_tab import render_dataset_chat_tab
from app.ui.automl_tab import render_automl_tab
from app.ui.explainability_tab import render_explainability_tab
from app.ui.prediction_tab import render_prediction_tab
from app.ui.drift_detection_tab import render_drift_detection_tab
from app.ui.reports_tab import render_reports_tab

# NEW
from app.auth.auth_guard import AuthGuard
from app.ui.notification_panel import render_notification_panel
from app.ui.profile_page import render_profile_page
from app.ui.history_page import render_history_page
from app.ui.settings_page import render_settings_page
from app.ui.activity_dashboard import render_activity_dashboard


# ===========================================================
# PAGE CONFIG
# ===========================================================

st.set_page_config(
    page_title="Enterprise AI Data Analyst Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================
# AUTHENTICATION
# ===========================================================

AuthGuard.protect()

# ===========================================================
# HEADER
# ===========================================================

st.title("🤖 Enterprise AI Data Analyst Copilot")
st.caption(
    "AI-Powered Business Intelligence, Data Analytics, Machine Learning & Predictive Intelligence Platform"
)

st.divider()

# ===========================================================
# SIDEBAR USER
# ===========================================================

with st.sidebar:

    st.success("✅ Logged In")

    user = AuthGuard.current_user()

    if user:

        try:
            st.write(f"👤 {user.email}")
        except Exception:
            pass

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):

        AuthGuard.logout()

    st.divider()

    render_notification_panel()

# ===========================================================
# FILE UPLOAD
# ===========================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel File",
    type=["csv", "xlsx", "xls"]
)

# ===========================================================
# NO DATASET
# ===========================================================

if uploaded_file is None:

    st.info("👆 Upload a CSV or Excel dataset to begin analysis.")

# ===========================================================
# DATASET AVAILABLE
# ===========================================================

else:

    try:

        df = FileLoader.load(uploaded_file)

        original_df = df.copy()

        cleaned_df, cleaning_summary = DataCleaningAgent.clean_data(df)

        profile = DatasetProfiler.profile(cleaned_df)

        render_sidebar(profile)

        eda_results = EDAAgent(cleaned_df).analyze()

        st.success("✅ Dataset Loaded Successfully!")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rows",
            profile.get("number_of_rows", len(cleaned_df))
        )

        c2.metric(
            "Columns",
            profile.get(
                "number_of_columns",
                len(cleaned_df.columns)
            )
        )

        c3.metric(
            "Numeric",
            len(profile.get("numeric_columns", []))
        )

        c4.metric(
            "Categorical",
            len(profile.get("categorical_columns", []))
        )

        st.divider()

        (
            tab_dashboard,
            tab_analytics,
            tab_ai,
            tab_chat,
            tab_automl,
            tab_explainability,
            tab_prediction,
            tab_drift,
            tab_reports,
            tab_profile,
            tab_history,
            tab_activity,
            tab_settings

        ) = st.tabs([

            "📊 Dashboard",
            "📈 Analytics",
            "🤖 AI Assistant",
            "💬 Chat",
            "🧠 AutoML",
            "🧩 Explainability",
            "🔮 Prediction",
            "📊 Drift Detection",
            "📄 Reports",
            "👤 Profile",
            "📜 History",
            "📈 Activity",
            "⚙️ Settings"

        ])

        # Dashboard

        with tab_dashboard:

            render_dashboard_tab(
                original_df,
                cleaned_df,
                profile,
                cleaning_summary
            )

        # Analytics

        with tab_analytics:

            render_analytics_tab(
                cleaned_df,
                eda_results
            )

        # AI

        with tab_ai:

            render_ai_tab(
                cleaned_df=cleaned_df,
                profile=profile,
                eda_results=eda_results
            )

        # Chat

        with tab_chat:

            render_dataset_chat_tab(
                cleaned_df
            )

        # AutoML

        with tab_automl:

            render_automl_tab(
                cleaned_df
            )

        # Explainability

        with tab_explainability:

            render_explainability_tab()

        # Prediction

        with tab_prediction:

            render_prediction_tab()

        # Drift

        with tab_drift:

            render_drift_detection_tab(
                original_df,
                cleaned_df
            )

        # Reports

        with tab_reports:

            render_reports_tab(
                profile,
                cleaning_summary,
                eda_results
            )

        # ==========================================================
        # PROFILE
        # ==========================================================

        with tab_profile:

            render_profile_page()

        # ==========================================================
        # HISTORY
        # ==========================================================

        with tab_history:

            render_history_page()

        # ==========================================================
        # ACTIVITY
        # ==========================================================

        with tab_activity:

            render_activity_dashboard()

        # ==========================================================
        # SETTINGS
        # ==========================================================

        with tab_settings:

            render_settings_page()

    except Exception as e:

        st.error(f"❌ Application Error: {e}")

        with st.expander("🔧 Error Details"):

            st.exception(e)
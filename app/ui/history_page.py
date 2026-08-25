import streamlit as st
import pandas as pd

from app.auth.auth_guard import AuthGuard
from app.database.database_service import DatabaseService
from app.auth.supabase_client import supabase


def render_history_page():

    st.title("📜 History Center")

    user = AuthGuard.current_user()

    if user is None:
        st.warning("Please login.")
        return

    email = user.email

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "💬 Chat History",
            "📄 Reports",
            "🔮 Predictions",
            "📂 Datasets"
        ]
    )

    # =====================================
    # CHAT HISTORY
    # =====================================

    with tab1:

        try:

            chats = DatabaseService.load_chat(email)

            if chats:

                st.dataframe(
                    pd.DataFrame(chats),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No chat history found.")

        except Exception as e:

            st.error(str(e))

    # =====================================
    # REPORTS
    # =====================================

    with tab2:

        try:

            reports = (
                supabase
                .table("reports")
                .select("*")
                .eq("user_email", email)
                .order("created_at", desc=True)
                .execute()
            )

            if reports.data:

                st.dataframe(
                    pd.DataFrame(reports.data),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No reports found.")

        except Exception as e:

            st.error(str(e))

    # =====================================
    # PREDICTIONS
    # =====================================

    with tab3:

        try:

            predictions = (
                supabase
                .table("predictions")
                .select("*")
                .eq("user_email", email)
                .order("created_at", desc=True)
                .execute()
            )

            if predictions.data:

                st.dataframe(
                    pd.DataFrame(predictions.data),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No predictions found.")

        except Exception as e:

            st.error(str(e))

    # =====================================
    # DATASETS
    # =====================================

    with tab4:

        try:

            datasets = (
                supabase
                .table("datasets")
                .select("*")
                .eq("user_email", email)
                .order("created_at", desc=True)
                .execute()
            )

            if datasets.data:

                st.dataframe(
                    pd.DataFrame(datasets.data),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No datasets found.")

        except Exception as e:

            st.error(str(e))
import streamlit as st
import pandas as pd

from app.auth.auth_guard import AuthGuard
from app.auth.supabase_client import supabase


def render_activity_dashboard():

    st.title("📊 Activity Dashboard")

    user = AuthGuard.current_user()

    if user is None:

        st.warning("Please login.")

        return

    email = user.email

    # =====================================================
    # COUNTS
    # =====================================================

    def count_rows(table_name):

        try:

            response = (
                supabase
                .table(table_name)
                .select("*")
                .eq("user_email", email)
                .execute()
            )

            return len(response.data)

        except Exception:

            return 0

    total_chats = count_rows("chat_history")

    total_reports = count_rows("reports")

    total_predictions = count_rows("predictions")

    total_datasets = count_rows("datasets")

    total_notifications = count_rows("notifications")

    # =====================================================
    # KPI CARDS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("💬 Chats", total_chats)

    c2.metric("📄 Reports", total_reports)

    c3.metric("🔮 Predictions", total_predictions)

    c4.metric("📂 Datasets", total_datasets)

    c5.metric("🔔 Notifications", total_notifications)

    st.divider()

    # =====================================================
    # RECENT ACTIVITY
    # =====================================================

    st.subheader("📜 Recent Activity")

    try:

        activity = (

            supabase

            .table("activity_logs")

            .select("*")

            .eq("user_email", email)

            .order("created_at", desc=True)

            .limit(20)

            .execute()

        )

        if activity.data:

            st.dataframe(

                pd.DataFrame(activity.data),

                use_container_width=True,

                hide_index=True

            )

        else:

            st.info("No activity yet.")

    except Exception as e:

        st.error(str(e))

    st.divider()

    # =====================================================
    # USER SUMMARY
    # =====================================================

    st.subheader("📈 Summary")

    summary = {

        "Chats": total_chats,

        "Reports": total_reports,

        "Predictions": total_predictions,

        "Datasets": total_datasets,

        "Notifications": total_notifications

    }

    st.json(summary)

    st.success("✅ Activity Dashboard Loaded Successfully")
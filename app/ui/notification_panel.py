import streamlit as st

from app.services.notification_service import NotificationService


def render_notification_panel():

    notifications = NotificationService.get_notifications()

    with st.expander("🔔 Notifications", expanded=False):

        if len(notifications) == 0:

            st.info("No notifications yet.")

            return

        for item in notifications:

            st.markdown(
                f"""
### {item['title']}

{item['message']}

🕒 {item['created_at']}

---
"""
            )
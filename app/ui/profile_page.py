import streamlit as st
from app.auth.auth_guard import AuthGuard


def render_profile_page():

    st.title("👤 User Profile")

    user = AuthGuard.current_user()

    if user is None:
        st.warning("User not logged in.")
        return

    st.subheader("Account Information")

    c1, c2 = st.columns(2)

    with c1:
        st.text_input(
            "Email",
            value=user.email,
            disabled=True
        )

    with c2:
        st.text_input(
            "User ID",
            value=user.id,
            disabled=True
        )

    st.divider()

    st.subheader("Statistics")

    a, b, c = st.columns(3)

    a.metric("Projects", "1")
    b.metric("Datasets", "0")
    c.metric("Reports", "0")

    st.divider()

    st.success("✅ Account connected with Supabase.")

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):
        AuthGuard.logout()
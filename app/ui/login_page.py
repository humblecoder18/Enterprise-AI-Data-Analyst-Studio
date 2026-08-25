import streamlit as st
from app.auth.auth_service import AuthService


def render_login_page():

    st.title("🔐 Enterprise AI Login")

    tab_login, tab_register = st.tabs(
        ["Login", "Register"]
    )

    # =====================================================
    # LOGIN
    # =====================================================

    with tab_login:

        email = st.text_input(
            "Email",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            use_container_width=True
        ):

            result = AuthService.sign_in(
                email,
                password
            )

            if result["success"]:

                st.session_state["logged_in"] = True
                st.session_state["user"] = result["user"]

                st.success(
                    "Login Successful!"
                )

                st.rerun()

            else:

                st.error(
                    result["message"]
                )

        st.divider()

        forgot_email = st.text_input(
            "Forgot Password Email",
            key="forgot_email"
        )

        if st.button(
            "Send Reset Link",
            use_container_width=True
        ):

            result = AuthService.reset_password(
                forgot_email
            )

            if result["success"]:
                st.success(
                    "Password reset email sent."
                )
            else:
                st.error(
                    result["message"]
                )

    # =====================================================
    # REGISTER
    # =====================================================

    with tab_register:

        reg_email = st.text_input(
            "Email",
            key="register_email"
        )

        reg_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password"
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            if reg_password != confirm:

                st.error(
                    "Passwords do not match."
                )

            else:

                result = AuthService.sign_up(
                    reg_email,
                    reg_password
                )

                if result["success"]:

                    st.success(
                        "Registration successful. Check your email for verification."
                    )

                else:

                    st.error(
                        result["message"]
                    )
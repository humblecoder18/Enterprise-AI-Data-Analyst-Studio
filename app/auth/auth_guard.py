import streamlit as st
from app.auth.session_manager import SessionManager
from app.ui.login_page import render_login_page


class AuthGuard:

    @staticmethod
    def protect():

        SessionManager.initialize()

        if not SessionManager.is_logged_in():

            render_login_page()

            st.stop()

    @staticmethod
    def logout():

        SessionManager.logout()

        st.rerun()

    @staticmethod
    def current_user():

        return SessionManager.current_user()
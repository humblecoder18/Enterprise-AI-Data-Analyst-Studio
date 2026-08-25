import streamlit as st
from app.auth.supabase_client import supabase


class SessionManager:

    @staticmethod
    def initialize():

        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False

        if "user" not in st.session_state:
            st.session_state.user = None

        try:

            response = supabase.auth.get_session()

            if (
                response
                and response.session
                and response.session.user
            ):

                st.session_state.logged_in = True
                st.session_state.user = response.session.user

        except Exception:
            pass

    @staticmethod
    def login(user):

        st.session_state.logged_in = True
        st.session_state.user = user

    @staticmethod
    def logout():

        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        st.session_state.logged_in = False
        st.session_state.user = None

    @staticmethod
    def is_logged_in():

        return st.session_state.get(
            "logged_in",
            False
        )

    @staticmethod
    def current_user():

        return st.session_state.get(
            "user",
            None
        )
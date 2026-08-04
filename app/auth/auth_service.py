from supabase import AuthApiError
from app.auth.supabase_client import supabase


class AuthService:

    @staticmethod
    def sign_up(email: str, password: str):

        try:

            response = supabase.auth.sign_up(
                {
                    "email": email,
                    "password": password
                }
            )

            return {
                "success": True,
                "user": response.user,
                "session": response.session
            }

        except Exception as e:

            return {
                "success": False,
                "message": str(e)
            }

    @staticmethod
    def sign_in(email: str, password: str):

        try:

            response = supabase.auth.sign_in_with_password(
                {
                    "email": email,
                    "password": password
                }
            )

            return {
                "success": True,
                "user": response.user,
                "session": response.session
            }

        except Exception as e:

            return {
                "success": False,
                "message": str(e)
            }

    @staticmethod
    def sign_out():

        try:

            supabase.auth.sign_out()

            return {
                "success": True
            }

        except Exception as e:

            return {
                "success": False,
                "message": str(e)
            }

    @staticmethod
    def reset_password(email):

        try:

            supabase.auth.reset_password_email(
                email
            )

            return {
                "success": True
            }

        except Exception as e:

            return {
                "success": False,
                "message": str(e)
            }

    @staticmethod
    def current_user():

        try:

            user = supabase.auth.get_user()

            return user

        except Exception:

            return None
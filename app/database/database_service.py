from datetime import datetime
from app.auth.supabase_client import supabase


class DatabaseService:
    supabase = supabase

    # ==========================================
    # SAVE CHAT
    # ==========================================

    @staticmethod
    def save_chat(user_email, question, answer):

        try:

            data = {
                "user_email": user_email,
                "question": question,
                "answer": answer,
                "created_at": datetime.utcnow().isoformat()
            }

            return (
                supabase
                .table("chat_history")
                .insert(data)
                .execute()
            )

        except Exception as e:

            print(e)

            return None

    # ==========================================
    # LOAD CHAT
    # ==========================================

    @staticmethod
    def load_chat(user_email):

        try:

            response = (
                supabase
                .table("chat_history")
                .select("*")
                .eq("user_email", user_email)
                .order("created_at", desc=True)
                .execute()
            )

            return response.data

        except Exception:

            return []

    # ==========================================
    # SAVE REPORT
    # ==========================================

    @staticmethod
    def save_report(user_email, report_name):

        try:

            return (
                supabase
                .table("reports")
                .insert({
                    "user_email": user_email,
                    "report_name": report_name,
                    "created_at": datetime.utcnow().isoformat()
                })
                .execute()
            )

        except Exception as e:

            print("SAVE REPORT ERROR:", e)

            raise

    # ==========================================
    # SAVE PREDICTION
    # ==========================================

    @staticmethod
    def save_prediction(user_email, prediction):

        try:

            return (
                supabase
                .table("predictions")
                .insert({
                    "user_email": user_email,
                    "prediction": str(prediction),
                    "created_at": datetime.utcnow().isoformat()
                })
                .execute()
            )

        except Exception:

            return None

    # ==========================================
    # SAVE NOTIFICATION
    # ==========================================

    @staticmethod
    def save_notification(user_email, title, message):

        try:

            return (
                supabase
                .table("notifications")
                .insert({
                    "user_email": user_email,
                    "title": title,
                    "message": message,
                    "created_at": datetime.utcnow().isoformat()
                })
                .execute()
            )

        except Exception:

            return None

    # ==========================================
    # LOAD NOTIFICATIONS
    # ==========================================

    @staticmethod
    def load_notifications(user_email):

        try:

            response = (
                supabase
                .table("notifications")
                .select("*")
                .eq("user_email", user_email)
                .order("created_at", desc=True)
                .execute()
            )

            return response.data

        except Exception:

            return []

    # ==========================================
    # SAVE DATASET
    # ==========================================

    @staticmethod
    def save_dataset(user_email, filename):

        try:

            return (
                supabase
                .table("datasets")
                .insert({
                    "user_email": user_email,
                    "dataset_name": filename,
                    "created_at": datetime.utcnow().isoformat()
                })
                .execute()
            )

        except Exception:

            return None

    # ==========================================
    # USER ACTIVITY
    # ==========================================

    @staticmethod
    def log_activity(user_email, activity):

        try:

            return (
                supabase
                .table("activity_logs")
                .insert({
                    "user_email": user_email,
                    "activity": activity,
                    "created_at": datetime.utcnow().isoformat()
                })
                .execute()
            )

        except Exception:

            return None
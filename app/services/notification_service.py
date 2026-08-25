from app.database.database_service import DatabaseService
from app.auth.auth_guard import AuthGuard


class NotificationService:

    @staticmethod
    def notify(title, message):

        try:

            user = AuthGuard.current_user()

            if user:

                DatabaseService.save_notification(
                    user.email,
                    title,
                    message
                )

        except Exception as e:

            print(e)

    @staticmethod
    def get_notifications():

        try:

            user = AuthGuard.current_user()

            if user:

                return DatabaseService.load_notifications(
                    user.email
                )

        except Exception:

            pass

        return []
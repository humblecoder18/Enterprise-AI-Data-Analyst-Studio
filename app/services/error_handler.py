import logging
import traceback
import streamlit as st

# =====================================================
# LOGGER CONFIGURATION
# =====================================================

logging.basicConfig(
    filename="enterprise_ai_errors.log",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class ErrorHandler:

    @staticmethod
    def handle_error(
        error: Exception,
        title="Something went wrong"
    ):

        logging.exception(title)

        st.error(f"❌ {title}")

        st.info(str(error))

    @staticmethod
    def handle_debug_error(
        error: Exception,
        title="Something went wrong"
    ):

        logging.exception(title)

        st.error(f"❌ {title}")

        st.info(str(error))

        with st.expander("Technical Details"):

            st.code(traceback.format_exc())
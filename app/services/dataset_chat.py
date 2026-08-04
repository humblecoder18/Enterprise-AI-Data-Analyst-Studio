import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from app.database.database_service import DatabaseService
from app.auth.auth_guard import AuthGuard
from app.services.ai_service import AIService


class DatasetChatService:
    """
    Enterprise AI Dataset Chat Service powered by Groq.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        model: str = "llama-3.3-70b-versatile",
        host: Optional[str] = None
    ):
        self.df = dataframe
        self.ai = AIService(model_name=model)
        self.chat_history: List[Dict] = []
        self.max_history = 3

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    def dataset_summary(self):
        summary = {
            "Rows": len(self.df),
            "Columns": len(self.df.columns),
            "Column Names": list(self.df.columns),
            "Data Types": {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            },
            "Missing Values": self.df.isnull().sum().to_dict(),
            "Numeric Columns": list(
                self.df.select_dtypes(
                    include=np.number
                ).columns
            ),
            "Categorical Columns": list(
                self.df.select_dtypes(
                    exclude=np.number
                ).columns
            )
        }
        return summary

    # =====================================================
    # DATA SAMPLE
    # =====================================================

    def sample_rows(self, rows: int = 3):
        return self.df.head(rows)

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    def add_to_history(self, role, content):
        self.chat_history.append({
            "role": role,
            "content": content
        })
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)

    def clear_history(self):
        self.chat_history = []

    # =====================================================
    # PROMPT
    # =====================================================

    def build_prompt(self, question: str):
        dataset_info = self.dataset_summary()
        preview = self.df.head(3).to_markdown()

        prompt = f"""
You are an Enterprise AI Data Analyst.

You are ONLY allowed to answer using the uploaded dataset.

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Dataset Preview (Top 3 rows):
{preview}

Conversation History (Last 3 messages):
{self.chat_history}

User Question:
{question}

Rules:
1. Answer only using the dataset.
2. If information is unavailable, say: "I cannot determine this from the uploaded dataset."
3. Be concise and professional. Limit your response to 350 words.
4. Use bullet points whenever possible.
"""
        return prompt

    # =====================================================
    # ASK AI
    # =====================================================

    def ask(self, question: str):
        self.add_to_history("user", question)
        prompt = self.build_prompt(question)

        try:
            answer = self.ai.ask(prompt, max_tokens=500)
        except Exception as e:
            answer = f"Unable to contact Groq.\n\n{str(e)}"

        self.add_to_history("assistant", answer)

        # ============================================
        # SAVE CHAT TO SUPABASE
        # ============================================
        try:
            user = AuthGuard.current_user()
            if user:
                DatabaseService.save_chat(
                    user.email,
                    question,
                    answer
                )
                DatabaseService.log_activity(
                    user.email,
                    "Asked a question in Dataset Chat"
                )
        except Exception as e:
            print(f"Database Error: {e}")

        return answer

    # =====================================================
    # BASIC DATASET STATISTICS
    # =====================================================

    def statistics(self):
        try:
            return self.df.describe(include="all")
        except Exception:
            return pd.DataFrame()

    # =====================================================
    # COLUMN INFORMATION
    # =====================================================

    def column_information(self):
        info = []
        for column in self.df.columns:
            info.append({
                "Column": column,
                "Type": str(self.df[column].dtype),
                "Missing Values": int(self.df[column].isnull().sum()),
                "Unique Values": int(self.df[column].nunique())
            })
        return pd.DataFrame(info)

    # =====================================================
    # SEARCH COLUMN
    # =====================================================

    def has_column(self, column_name):
        return column_name in self.df.columns

    # =====================================================
    # GET COLUMN
    # =====================================================

    def get_column(self, column_name):
        if self.has_column(column_name):
            return self.df[column_name]
        return None

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    def numeric_columns(self):
        return list(
            self.df.select_dtypes(
                include=np.number
            ).columns
        )

    # =====================================================
    # CATEGORICAL COLUMNS
    # =====================================================

    def categorical_columns(self):
        return list(
            self.df.select_dtypes(
                exclude=np.number
            ).columns
        )

    # =====================================================
    # SAMPLE DATA
    # =====================================================

    def sample(self, n=3):
        return self.df.head(n)

    # =====================================================
    # SHAPE
    # =====================================================

    def shape(self):
        return self.df.shape

    # =====================================================
    # MISSING VALUES
    # =====================================================

    def missing_values(self):
        return self.df.isnull().sum()

    # =====================================================
    # DUPLICATES
    # =====================================================

    def duplicates(self):
        return int(self.df.duplicated().sum())

    # =====================================================
    # CORRELATION MATRIX
    # =====================================================

    def correlation_matrix(self):
        try:
            return self.df.corr(numeric_only=True)
        except Exception:
            return pd.DataFrame()

    # =====================================================
    # TOP ROWS
    # =====================================================

    def top_rows(self, rows: int = 3):
        return self.df.head(rows)

    # =====================================================
    # BOTTOM ROWS
    # =====================================================

    def bottom_rows(self, rows: int = 3):
        return self.df.tail(rows)

    # =====================================================
    # RANDOM SAMPLE
    # =====================================================

    def random_sample(self, rows: int = 3):
        rows = min(rows, len(self.df))
        return self.df.sample(rows, random_state=42)

    # =====================================================
    # VALUE COUNTS
    # =====================================================

    def value_counts(self, column: str):
        if column not in self.df.columns:
            return None
        return self.df[column].value_counts()

    # =====================================================
    # UNIQUE VALUES
    # =====================================================

    def unique_values(self, column: str):
        if column not in self.df.columns:
            return []
        return list(self.df[column].dropna().unique())

    # =====================================================
    # DATASET REPORT
    # =====================================================

    def dataset_report(self):
        report = {
            "Rows": len(self.df),
            "Columns": len(self.df.columns),
            "Missing Values": int(self.df.isnull().sum().sum()),
            "Duplicate Rows": int(self.df.duplicated().sum()),
            "Numeric Columns": self.numeric_columns(),
            "Categorical Columns": self.categorical_columns()
        }
        return report

    # =====================================================
    # AI DATASET SUMMARY
    # =====================================================

    def generate_summary(self):
        prompt = f"""
You are an expert Data Scientist.

Generate a professional summary for this dataset.

Dataset Information:
{json.dumps(self.dataset_report(), indent=2)}

Keep the answer under 200 words.
"""
        try:
            return self.ai.ask(prompt, max_tokens=300)
        except Exception as e:
            return str(e)

    # =====================================================
    # AI INSIGHTS
    # =====================================================

    def generate_insights(self):
        prompt = f"""
You are a Senior Business Analyst.

Based on this dataset:
{json.dumps(self.dataset_report(), indent=2)}

Generate:
• Key observations
• Business insights
• Data quality issues
• Recommended ML tasks

Keep the answer under 350 words.
"""
        try:
            return self.ai.ask(prompt, max_tokens=400)
        except Exception as e:
            return str(e)

    # =====================================================
    # SUGGESTED QUESTIONS
    # =====================================================

    def suggested_questions(self):
        return [
            "Summarize this dataset.",
            "What are the important columns?",
            "Are there missing values?",
            "Which columns are numeric?",
            "Which columns are categorical?",
            "Suggest suitable ML models.",
            "Recommend feature engineering.",
            "Explain this dataset."
        ]

    # =====================================================
    # RESET CHAT
    # =====================================================

    def reset_chat(self):
        self.chat_history = []

    # =====================================================
    # EXPORT CHAT
    # =====================================================

    def export_chat(self):
        return pd.DataFrame(self.chat_history)

    # =====================================================
    # VERSION
    # =====================================================
    @staticmethod
    def version():
        return "Enterprise Dataset Chat v1.0"
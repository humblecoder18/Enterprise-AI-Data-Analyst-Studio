import re
import pandas as pd

from app.services.ai_service import AIService


class ChatAgent:
    """
    Hybrid dataset chat agent.

    Pandas performs calculations on the complete dataset.
    Groq is used to interpret questions and explain results.

    This avoids asking the LLM to calculate statistics from
    only a small dataset preview.
    """

    def __init__(self):
        self.ai = AIService()

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    @staticmethod
    def _normalize(text):
        return re.sub(
            r"[^a-z0-9]",
            "",
            str(text).lower()
        )

    # =========================================================
    # FIND COLUMN
    # =========================================================

    def _find_column(self, df, keywords):
        """
        Try to locate a dataframe column using possible
        words used by the user.
        """

        normalized_columns = {
            self._normalize(column): column
            for column in df.columns
        }

        # Exact normalized match
        for keyword in keywords:

            normalized_keyword = self._normalize(keyword)

            if normalized_keyword in normalized_columns:
                return normalized_columns[
                    normalized_keyword
                ]

        # Partial match
        for keyword in keywords:

            normalized_keyword = self._normalize(keyword)

            for normalized_column, original_column in (
                normalized_columns.items()
            ):

                if (
                    normalized_keyword
                    and normalized_keyword
                    in normalized_column
                ):
                    return original_column

        return None

    # =========================================================
    # BASIC DATASET SUMMARY
    # =========================================================

    def _dataset_summary(self, df):

        numeric_df = df.select_dtypes(
            include="number"
        )

        categorical_df = df.select_dtypes(
            exclude="number"
        )

        summary_parts = [
            f"Rows: {df.shape[0]}",
            f"Columns: {df.shape[1]}",
            f"Column names: {list(df.columns)}",
            (
                "Numeric columns: "
                f"{list(numeric_df.columns)}"
            ),
            (
                "Categorical columns: "
                f"{list(categorical_df.columns)}"
            ),
            (
                "Total missing values: "
                f"{int(df.isnull().sum().sum())}"
            )
        ]

        return "\n".join(summary_parts)

    # =========================================================
    # COLUMN STATISTICS
    # =========================================================

    def _column_statistics(self, df):

        results = []

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:

            series = df[column].dropna()

            if series.empty:
                continue

            results.append(
                {
                    "column": column,
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "min": float(series.min()),
                    "max": float(series.max())
                }
            )

        return results

    # =========================================================
    # DETECT COMMON BUSINESS QUESTIONS
    # =========================================================

    def _perform_direct_analysis(
        self,
        df,
        question
    ):
        """
        Perform common analytical calculations directly
        using the complete dataframe.

        Returns:
            str | None
        """

        q = question.lower()

        # =====================================================
        # DATASET SIZE
        # =====================================================

        if (
            "how many rows" in q
            or "number of rows" in q
            or "how many records" in q
            or "how many employees" in q
        ):

            return (
                f"The dataset contains "
                f"{df.shape[0]} records."
            )

        if (
            "how many columns" in q
            or "number of columns" in q
        ):

            return (
                f"The dataset contains "
                f"{df.shape[1]} columns."
            )

        # =====================================================
        # MISSING VALUES
        # =====================================================

        if (
            "missing value" in q
            or "null value" in q
            or "missing data" in q
        ):

            missing = (
                df.isnull()
                .sum()
            )

            missing = missing[
                missing > 0
            ]

            if missing.empty:

                return (
                    "There are no missing values "
                    "in the dataset."
                )

            return (
                "Missing values by column:\n"
                + missing.to_string()
            )

        # =====================================================
        # ATTRITION QUESTIONS
        # =====================================================

        attrition_col = self._find_column(
            df,
            ["attrition"]
        )

        if (
            attrition_col
            and "attrition" in q
        ):

            counts = (
                df[attrition_col]
                .value_counts(dropna=False)
            )

            percentages = (
                df[attrition_col]
                .value_counts(
                    normalize=True,
                    dropna=False
                )
                * 100
            )

            result = (
                "Attrition distribution:\n\n"
            )

            for value in counts.index:

                result += (
                    f"{value}: "
                    f"{int(counts[value])} employees "
                    f"({percentages[value]:.2f}%)\n"
                )

            return result

        # =====================================================
        # DEPARTMENT + SALARY / INCOME
        # =====================================================

        department_col = self._find_column(
            df,
            ["department"]
        )

        income_col = self._find_column(
            df,
            [
                "monthlyincome",
                "monthly income",
                "salary",
                "income"
            ]
        )

        if (
            department_col
            and income_col
            and "department" in q
            and (
                "salary" in q
                or "income" in q
            )
        ):

            grouped = (
                df.groupby(
                    department_col
                )[income_col]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            if grouped.empty:
                return None

            if (
                "highest" in q
                or "maximum" in q
                or "max " in q
                or "most" in q
            ):

                department = grouped.index[0]
                value = grouped.iloc[0]

                return (
                    f"The department with the highest "
                    f"average {income_col} is "
                    f"{department}, with an average of "
                    f"{value:,.2f}.\n\n"
                    f"Department averages:\n"
                    f"{grouped.to_string()}"
                )

            if (
                "lowest" in q
                or "minimum" in q
                or "min " in q
            ):

                department = grouped.index[-1]
                value = grouped.iloc[-1]

                return (
                    f"The department with the lowest "
                    f"average {income_col} is "
                    f"{department}, with an average of "
                    f"{value:,.2f}.\n\n"
                    f"Department averages:\n"
                    f"{grouped.to_string()}"
                )

            return (
                f"Average {income_col} by "
                f"{department_col}:\n\n"
                f"{grouped.to_string()}"
            )

        # =====================================================
        # DEPARTMENT EMPLOYEE COUNTS
        # =====================================================

        if (
            department_col
            and "department" in q
            and (
                "how many" in q
                or "count" in q
                or "employees" in q
            )
        ):

            counts = (
                df[department_col]
                .value_counts()
            )

            return (
                "Employee count by department:\n\n"
                + counts.to_string()
            )

        # =====================================================
        # GENDER DISTRIBUTION
        # =====================================================

        gender_col = self._find_column(
            df,
            ["gender"]
        )

        if (
            gender_col
            and "gender" in q
        ):

            counts = (
                df[gender_col]
                .value_counts()
            )

            percentages = (
                df[gender_col]
                .value_counts(
                    normalize=True
                )
                * 100
            )

            result = (
                "Gender distribution:\n\n"
            )

            for value in counts.index:

                result += (
                    f"{value}: "
                    f"{int(counts[value])} "
                    f"({percentages[value]:.2f}%)\n"
                )

            return result

        # =====================================================
        # OVERTIME
        # =====================================================

        overtime_col = self._find_column(
            df,
            ["overtime", "over time"]
        )

        if (
            overtime_col
            and "overtime" in q.replace(" ", "")
        ):

            counts = (
                df[overtime_col]
                .value_counts()
            )

            percentages = (
                df[overtime_col]
                .value_counts(
                    normalize=True
                )
                * 100
            )

            result = (
                "Overtime distribution:\n\n"
            )

            for value in counts.index:

                result += (
                    f"{value}: "
                    f"{int(counts[value])} employees "
                    f"({percentages[value]:.2f}%)\n"
                )

            return result

        # =====================================================
        # AGE
        # =====================================================

        age_col = self._find_column(
            df,
            ["age"]
        )

        if age_col and "age" in q:

            series = df[age_col].dropna()

            if series.empty:
                return None

            if (
                "average" in q
                or "mean" in q
            ):

                return (
                    f"The average {age_col} is "
                    f"{series.mean():.2f}."
                )

            if (
                "oldest" in q
                or "maximum" in q
                or "highest" in q
            ):

                return (
                    f"The maximum {age_col} is "
                    f"{series.max()}."
                )

            if (
                "youngest" in q
                or "minimum" in q
                or "lowest" in q
            ):

                return (
                    f"The minimum {age_col} is "
                    f"{series.min()}."
                )

        # =====================================================
        # MONTHLY INCOME
        # =====================================================

        if (
            income_col
            and (
                "salary" in q
                or "income" in q
            )
        ):

            series = (
                df[income_col]
                .dropna()
            )

            if not series.empty:

                if (
                    "average" in q
                    or "mean" in q
                ):

                    return (
                        f"The average {income_col} "
                        f"across the dataset is "
                        f"{series.mean():,.2f}."
                    )

                if (
                    "highest" in q
                    or "maximum" in q
                ):

                    return (
                        f"The highest {income_col} "
                        f"is {series.max():,.2f}."
                    )

                if (
                    "lowest" in q
                    or "minimum" in q
                ):

                    return (
                        f"The lowest {income_col} "
                        f"is {series.min():,.2f}."
                    )

        return None

    # =========================================================
    # BUILD ANALYTICAL CONTEXT
    # =========================================================

    def _build_context(
        self,
        df,
        question
    ):
        """
        Build a compact but useful representation of the
        complete dataset for questions not handled directly.
        """

        parts = []

        # Dataset structure
        parts.append(
            self._dataset_summary(df)
        )

        # Missing values per column
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if not missing_cols.empty:
            parts.append(
                "\nMissing Values per Column:\n"
                + missing_cols.to_dict().__str__()
            )

        # Key correlations (above 0.5 or below -0.5)
        try:
            corr = df.corr(numeric_only=True)
            high_corr = []
            for col in corr.columns:
                for idx in corr.index:
                    if col != idx and abs(corr.loc[idx, col]) > 0.5:
                        pair = sorted([col, idx])
                        val = corr.loc[idx, col]
                        desc = f"{pair[0]} and {pair[1]}: {val:.2f}"
                        if desc not in high_corr:
                            high_corr.append(desc)
            if high_corr:
                parts.append(
                    "\nKey Correlations (abs > 0.5):\n"
                    + "\n".join(high_corr[:10])
                )
        except Exception:
            pass

        # Small sample for understanding structure (limited to 3 rows)
        try:
            preview = (
                df.head(3)
                .to_string(
                    index=False
                )
            )

            parts.append(
                "\nExample Rows (Top 3):\n"
                + preview
            )

        except Exception:
            pass

        return "\n".join(parts)

    # =========================================================
    # ASK
    # =========================================================

    def ask(
        self,
        df: pd.DataFrame,
        question: str
    ):
        """
        Answer a natural-language question about a dataframe.
        """

        if df is None or df.empty:

            return (
                "The dataset is empty, so I cannot "
                "perform the requested analysis."
            )

        if not question or not question.strip():

            return (
                "Please provide a question about "
                "the dataset."
            )

        # =====================================================
        # FIRST: TRY DIRECT PANDAS ANALYSIS
        # =====================================================

        direct_result = (
            self._perform_direct_analysis(
                df,
                question
            )
        )

        if direct_result:

            prompt = f"""
You are a Senior Data Analyst.

The following result was calculated directly from the COMPLETE
dataset using pandas.

User Question:
{question}

Verified Calculation:
{direct_result}

Rules:
- Answer the user's question directly.
- Use the verified calculation above.
- Do not change any calculated numbers.
- Do not invent additional statistics.
- Keep the answer concise and professional.
- Mention the key result first.
- You may briefly explain what the result means.
"""

            try:

                return self.ai.ask(
                    prompt,
                    max_tokens=400
                )

            except Exception:

                # If Groq fails, the verified pandas
                # result is still useful.
                return direct_result

        # =====================================================
        # SECOND: BUILD FULL DATASET SUMMARY
        # =====================================================

        context = self._build_context(
            df,
            question
        )

        prompt = f"""
You are a Senior Data Analyst working inside an
Enterprise AI Data Analyst Copilot.

Answer the user's question using ONLY the analytical
dataset information supplied below.

IMPORTANT:
The statistics and value counts were calculated from
the COMPLETE dataset using pandas.

Dataset Analysis:
{context}

User Question:
{question}

Rules:

1. Answer only from the supplied dataset analysis.

2. Do not invent values or statistics.

3. The numeric statistics represent the complete dataset.

4. The categorical value counts represent the complete dataset.

5. Example rows are included only to understand the structure
   of the dataset. Do not treat them as the entire dataset.

6. If the available analysis is insufficient to answer the
   question accurately, clearly say that the requested
   calculation is not currently available.

7. Give the direct answer first.

8. Keep the response concise, professional, and
   business-friendly.

9. When appropriate, explain what the result could mean
   from a business or analytical perspective.
"""

        return self.ai.ask(
            prompt,
            max_tokens=450
        )
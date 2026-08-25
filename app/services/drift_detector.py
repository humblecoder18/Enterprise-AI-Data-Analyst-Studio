import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency


class DriftDetector:
    """
    Enterprise Data Drift Detection

    Supports

    • Numeric Drift
    • Categorical Drift
    • Missing Value Drift
    • Distribution Drift
    """

    def __init__(

        self,

        original_df: pd.DataFrame,

        cleaned_df: pd.DataFrame

    ):

        self.original = original_df.copy()

        self.cleaned = cleaned_df.copy()

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    def numeric_columns(self):

        return list(

            self.original.select_dtypes(

                include=np.number

            ).columns

        )

    # =====================================================
    # CATEGORICAL COLUMNS
    # =====================================================

    def categorical_columns(self):

        return list(

            self.original.select_dtypes(

                exclude=np.number

            ).columns

        )
        # =====================================================
    # NUMERIC DRIFT
    # =====================================================

    def numeric_drift(self):

        results = []

        for column in self.numeric_columns():

            if column not in self.cleaned.columns:
                continue

            original = (
                self.original[column]
                .dropna()
            )

            cleaned = (
                self.cleaned[column]
                .dropna()
            )

            if len(original) == 0 or len(cleaned) == 0:
                continue

            statistic, p_value = ks_2samp(
                original,
                cleaned
            )

            if p_value > 0.05:
                status = "✅ Stable"
            elif p_value > 0.01:
                status = "⚠ Moderate Drift"
            else:
                status = "❌ High Drift"

            results.append({

                "Column": column,

                "Type": "Numeric",

                "KS Statistic": round(
                    statistic,
                    4
                ),

                "P-Value": round(
                    p_value,
                    4
                ),

                "Status": status

            })

        return pd.DataFrame(results)

    # =====================================================
    # CATEGORICAL DRIFT
    # =====================================================

    def categorical_drift(self):

        results = []

        for column in self.categorical_columns():

            if column not in self.cleaned.columns:
                continue

            original_counts = (
                self.original[column]
                .fillna("Missing")
                .value_counts()
            )

            cleaned_counts = (
                self.cleaned[column]
                .fillna("Missing")
                .value_counts()
            )

            categories = sorted(

                set(original_counts.index)

                |

                set(cleaned_counts.index)

            )

            original_freq = [

                original_counts.get(cat, 0)

                for cat in categories

            ]

            cleaned_freq = [

                cleaned_counts.get(cat, 0)

                for cat in categories

            ]

            contingency = np.array([
                original_freq,
                cleaned_freq
            ])

            try:

                chi2, p_value, _, _ = chi2_contingency(
                    contingency
                )

            except Exception:

                p_value = 1.0

            if p_value > 0.05:
                status = "✅ Stable"
            elif p_value > 0.01:
                status = "⚠ Moderate Drift"
            else:
                status = "❌ High Drift"

            results.append({

                "Column": column,

                "Type": "Categorical",

                "P-Value": round(
                    p_value,
                    4
                ),

                "Status": status

            })

        return pd.DataFrame(results)

    # =====================================================
    # MISSING VALUE DRIFT
    # =====================================================

    def missing_value_drift(self):

        results = []

        columns = sorted(

            set(self.original.columns)

            &

            set(self.cleaned.columns)

        )

        for column in columns:

            original_missing = int(
                self.original[column]
                .isnull()
                .sum()
            )

            cleaned_missing = int(
                self.cleaned[column]
                .isnull()
                .sum()
            )

            difference = abs(

                original_missing

                -

                cleaned_missing

            )

            results.append({

                "Column": column,

                "Original Missing":
                    original_missing,

                "Cleaned Missing":
                    cleaned_missing,

                "Difference":
                    difference

            })

        return pd.DataFrame(results)
        # =====================================================
    # OVERALL DRIFT SCORE
    # =====================================================

    def overall_drift_score(self):

        numeric = self.numeric_drift()

        categorical = self.categorical_drift()

        total_columns = len(numeric) + len(categorical)

        if total_columns == 0:

            return 100.0

        stable = 0

        moderate = 0

        high = 0

        for _, row in numeric.iterrows():

            if "Stable" in row["Status"]:
                stable += 1

            elif "Moderate" in row["Status"]:
                moderate += 1

            else:
                high += 1

        for _, row in categorical.iterrows():

            if "Stable" in row["Status"]:
                stable += 1

            elif "Moderate" in row["Status"]:
                moderate += 1

            else:
                high += 1

        score = (
            (
                stable
                + (0.5 * moderate)
            )
            / total_columns
        ) * 100

        return round(score, 2)

    # =====================================================
    # RISK LEVEL
    # =====================================================

    def risk_level(self):

        score = self.overall_drift_score()

        if score >= 90:

            return "🟢 Low Risk"

        elif score >= 70:

            return "🟡 Moderate Risk"

        return "🔴 High Risk"

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self):

        numeric = self.numeric_drift()

        categorical = self.categorical_drift()

        missing = self.missing_value_drift()

        return {

            "Overall Drift Score":

                self.overall_drift_score(),

            "Risk":

                self.risk_level(),

            "Numeric Columns":

                len(numeric),

            "Categorical Columns":

                len(categorical),

            "Missing Value Report":

                len(missing)

        }

    # =====================================================
    # COMPLETE REPORT
    # =====================================================

    def full_report(self):

        return {

            "summary":

                self.summary(),

            "numeric":

                self.numeric_drift(),

            "categorical":

                self.categorical_drift(),

            "missing":

                self.missing_value_drift()

        }
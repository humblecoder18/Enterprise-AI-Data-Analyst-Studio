from datetime import datetime
import html
import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


class PDFReportGenerator:
    """
    Professional PDF report generator for the
    Enterprise AI Data Analyst Copilot.
    """

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _format_value(value):

        if value is None:
            return "N/A"

        try:
            if isinstance(value, float):

                if math.isnan(value):
                    return "N/A"

                return f"{value:,.4f}"

            if isinstance(value, int):
                return f"{value:,}"

        except Exception:
            pass

        return str(value)

    @staticmethod
    def _clean_text(text):

        if not text:
            return "Not available."

        text = str(text)

        text = html.escape(text)

        text = text.replace("### ", "")
        text = text.replace("## ", "")
        text = text.replace("# ", "")
        text = text.replace("**", "")

        text = text.replace(
            "\n",
            "<br/>"
        )

        return text

    @staticmethod
    def _pretty_name(value):

        if value is None:
            return "N/A"

        return (
            str(value)
            .replace("_", " ")
            .replace("__", " ")
            .strip()
            .title()
        )

    # =========================================================
    # FOOTER
    # =========================================================

    @staticmethod
    def _add_page_number(canvas, doc):

        canvas.saveState()

        width, _ = A4

        canvas.setFont(
            "Helvetica",
            8
        )

        canvas.setFillColor(
            colors.grey
        )

        canvas.drawString(
            40,
            25,
            "Enterprise AI Data Analyst Copilot"
        )

        canvas.drawRightString(
            width - 40,
            25,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    # =========================================================
    # GENERATE REPORT
    # =========================================================

    @staticmethod
    def generate(
        output_path: str,
        profile: dict,
        cleaning_summary: dict,
        eda_results: dict,
        ai_insights: str,
        ml_recommendations: str,
        automl_results=None,
        prediction_result=None,
        prediction_input=None,
        local_explanation=None,
        global_explanation=None,
        ai_business_explanation=None,
    ):

        # =====================================================
        # SAFE DEFAULTS
        # =====================================================

        profile = profile or {}
        cleaning_summary = cleaning_summary or {}
        eda_results = eda_results or {}

        automl_results = automl_results or {}
        prediction_result = prediction_result or {}
        prediction_input = prediction_input or {}

        local_explanation = local_explanation or {}
        global_explanation = global_explanation or {}

        # =====================================================
        # DOCUMENT
        # =====================================================

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=45,
            leftMargin=45,
            topMargin=50,
            bottomMargin=50,
            title=(
                "Enterprise AI Data Analyst "
                "Copilot Report"
            ),
        )

        styles = getSampleStyleSheet()

        # =====================================================
        # STYLES
        # =====================================================

        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#17365D"
            ),
            spaceAfter=12,
        )

        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#4F81BD"
            ),
            spaceAfter=20,
        )

        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading1"],
            fontSize=15,
            leading=19,
            textColor=colors.HexColor(
                "#17365D"
            ),
            spaceBefore=10,
            spaceAfter=12,
        )

        subheading_style = ParagraphStyle(
            "SubHeading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor(
                "#365F91"
            ),
            spaceBefore=8,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            spaceAfter=7,
        )

        small_style = ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
        )

        elements = []

        # =====================================================
        # COVER
        # =====================================================

        elements.append(
            Paragraph(
                "Enterprise AI Data Analyst Copilot",
                title_style
            )
        )

        elements.append(
            Paragraph(
                (
                    "Business Intelligence, AutoML & "
                    "Predictive Analytics Report"
                ),
                subtitle_style
            )
        )

        elements.append(
            Spacer(
                1,
                0.15 * inch
            )
        )

        generated_time = (
            datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            )
        )

        elements.append(
            Paragraph(
                (
                    f"<b>Generated:</b> "
                    f"{generated_time}"
                ),
                body_style
            )
        )

        elements.append(
            Spacer(
                1,
                0.2 * inch
            )
        )

        # =====================================================
        # BASIC DATASET INFORMATION
        # =====================================================

        rows = profile.get(
            "number_of_rows",
            0
        )

        columns = profile.get(
            "number_of_columns",
            0
        )

        numeric_columns = profile.get(
            "numeric_columns",
            []
        )

        categorical_columns = profile.get(
            "categorical_columns",
            []
        )

        missing_values = profile.get(
            "missing_values_per_column",
            {}
        )

        duplicate_rows = profile.get(
            "duplicate_row_count",
            0
        )

        try:
            total_missing = sum(
                int(value)
                for value
                in missing_values.values()
            )

        except Exception:
            total_missing = 0

        # =====================================================
        # 1. EXECUTIVE SUMMARY
        # =====================================================

        elements.append(
            Paragraph(
                "1. Executive Summary",
                heading_style
            )
        )

        summary_text = (
            f"This automated analysis evaluated a dataset "
            f"containing <b>{rows:,} rows</b> and "
            f"<b>{columns:,} columns</b>. "
            f"The dataset contains "
            f"<b>{len(numeric_columns)} numerical</b> and "
            f"<b>{len(categorical_columns)} categorical</b> "
            f"features. Profiling identified "
            f"<b>{total_missing:,} missing values</b> and "
            f"<b>{duplicate_rows:,} duplicate rows</b>."
        )

        if automl_results:

            target = automl_results.get(
                "target_column",
                "N/A"
            )

            best_model = automl_results.get(
                "best_model",
                "N/A"
            )

            summary_text += (
                f" AutoML analysis was performed for "
                f"<b>{html.escape(str(target))}</b>, "
                f"with <b>{html.escape(str(best_model))}</b> "
                f"selected as the best-performing model."
            )

        if prediction_result:

            predicted_value = (
                prediction_result.get(
                    "prediction",
                    "N/A"
                )
            )

            summary_text += (
                f" The trained model was also used to "
                f"generate a prediction of "
                f"<b>{html.escape(str(predicted_value))}</b> "
                f"for the supplied prediction input."
            )

        elements.append(
            Paragraph(
                summary_text,
                body_style
            )
        )

        # =====================================================
        # 2. DATASET OVERVIEW
        # =====================================================

        elements.append(
            Paragraph(
                "2. Dataset Overview",
                heading_style
            )
        )

        overview_data = [
            [
                "Metric",
                "Value"
            ],
            [
                "Total Rows",
                f"{rows:,}"
            ],
            [
                "Total Columns",
                f"{columns:,}"
            ],
            [
                "Numeric Columns",
                str(
                    len(
                        numeric_columns
                    )
                )
            ],
            [
                "Categorical Columns",
                str(
                    len(
                        categorical_columns
                    )
                )
            ],
            [
                "Missing Values",
                f"{total_missing:,}"
            ],
            [
                "Duplicate Rows",
                f"{duplicate_rows:,}"
            ],
        ]

        overview_table = Table(
            overview_data,
            colWidths=[
                3.2 * inch,
                2.2 * inch
            ],
            repeatRows=1,
        )

        overview_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#17365D"
                        )
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    ),
                ]
            )
        )

        elements.append(
            overview_table
        )

        elements.append(
            Spacer(
                1,
                0.2 * inch
            )
        )

        # =====================================================
        # 3. DATA CLEANING
        # =====================================================

        elements.append(
            Paragraph(
                "3. Data Quality & Cleaning",
                heading_style
            )
        )

        if cleaning_summary:

            cleaning_data = [
                [
                    "Cleaning Operation",
                    "Result"
                ]
            ]

            for key, value in (
                cleaning_summary.items()
            ):

                cleaning_data.append(
                    [
                        PDFReportGenerator
                        ._pretty_name(
                            key
                        ),

                        PDFReportGenerator
                        ._format_value(
                            value
                        )
                    ]
                )

            cleaning_table = Table(
                cleaning_data,
                colWidths=[
                    3.2 * inch,
                    2.2 * inch
                ],
                repeatRows=1,
            )

            cleaning_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor(
                                "#4F81BD"
                            )
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold"
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                    ]
                )
            )

            elements.append(
                cleaning_table
            )

        else:

            elements.append(
                Paragraph(
                    (
                        "No cleaning summary "
                        "was available."
                    ),
                    body_style
                )
            )

        # =====================================================
        # 4. EXPLORATORY DATA ANALYSIS
        # =====================================================

        elements.append(
            Paragraph(
                "4. Exploratory Data Analysis",
                heading_style
            )
        )

        elements.append(
            Paragraph(
                "4.1 Dataset Health",
                subheading_style
            )
        )

        health_data = [
            [
                "Metric",
                "Result"
            ],
            [
                "Rows Analyzed",
                f"{rows:,}"
            ],
            [
                "Columns Analyzed",
                f"{columns:,}"
            ],
            [
                "Missing Values",
                f"{total_missing:,}"
            ],
            [
                "Duplicate Rows",
                f"{duplicate_rows:,}"
            ],
        ]

        health_table = Table(
            health_data,
            colWidths=[
                3.2 * inch,
                2.2 * inch
            ]
        )

        health_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#D9EAF7"
                        )
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                ]
            )
        )

        elements.append(
            health_table
        )

        # =====================================================
        # 4.2 NUMERICAL STATISTICS
        # =====================================================

        elements.append(
            Paragraph(
                "4.2 Key Numerical Statistics",
                subheading_style
            )
        )

        summary_statistics = (
            eda_results.get(
                "summary_statistics",
                {}
            )
        )

        means = summary_statistics.get(
            "mean",
            {}
        )

        medians = summary_statistics.get(
            "median",
            {}
        )

        minimums = summary_statistics.get(
            "min",
            {}
        )

        maximums = summary_statistics.get(
            "max",
            {}
        )

        if means:

            selected_features = list(
                means.keys()
            )[:10]

            statistics_data = [
                [
                    "Feature",
                    "Mean",
                    "Median",
                    "Min",
                    "Max"
                ]
            ]

            for feature in selected_features:

                statistics_data.append(
                    [
                        PDFReportGenerator
                        ._pretty_name(
                            feature
                        ),

                        PDFReportGenerator
                        ._format_value(
                            means.get(
                                feature
                            )
                        ),

                        PDFReportGenerator
                        ._format_value(
                            medians.get(
                                feature
                            )
                        ),

                        PDFReportGenerator
                        ._format_value(
                            minimums.get(
                                feature
                            )
                        ),

                        PDFReportGenerator
                        ._format_value(
                            maximums.get(
                                feature
                            )
                        ),
                    ]
                )

            statistics_table = Table(
                statistics_data,
                colWidths=[
                    2.0 * inch,
                    0.9 * inch,
                    0.9 * inch,
                    0.8 * inch,
                    0.8 * inch,
                ],
                repeatRows=1,
            )

            statistics_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor(
                                "#4F81BD"
                            )
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold"
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.grey
                        ),
                        (
                            "FONTSIZE",
                            (0, 0),
                            (-1, -1),
                            8
                        ),
                    ]
                )
            )

            elements.append(
                statistics_table
            )

        else:

            elements.append(
                Paragraph(
                    (
                        "No numerical summary "
                        "statistics were available."
                    ),
                    body_style
                )
            )

        # =====================================================
        # AI SECTION
        # =====================================================

        elements.append(
            PageBreak()
        )

        # =====================================================
        # 5. AI INSIGHTS
        # =====================================================

        elements.append(
            Paragraph(
                "5. AI-Generated Insights",
                heading_style
            )
        )

        elements.append(
            Paragraph(
                PDFReportGenerator
                ._clean_text(
                    ai_insights
                ),
                body_style
            )
        )

        # =====================================================
        # 6. ML RECOMMENDATIONS
        # =====================================================

        elements.append(
            Paragraph(
                "6. Machine Learning Recommendations",
                heading_style
            )
        )

        elements.append(
            Paragraph(
                PDFReportGenerator
                ._clean_text(
                    ml_recommendations
                ),
                body_style
            )
        )

        # =====================================================
        # 7. AUTOML ANALYSIS
        # =====================================================

        elements.append(
            PageBreak()
        )

        elements.append(
            Paragraph(
                "7. AutoML Model Analysis",
                heading_style
            )
        )

        if automl_results:

            target_column = (
                automl_results.get(
                    "target_column",
                    "N/A"
                )
            )

            problem_type = (
                automl_results.get(
                    "problem_type",
                    "N/A"
                )
            )

            best_model = (
                automl_results.get(
                    "best_model",
                    "N/A"
                )
            )

            selection_metric = (
                automl_results.get(
                    "selection_metric",
                    "N/A"
                )
            )

            selection_score = (
                automl_results.get(
                    "selection_score",
                    "N/A"
                )
            )

            automl_data = [
                [
                    "AutoML Property",
                    "Result"
                ],
                [
                    "Target Column",
                    str(
                        target_column
                    )
                ],
                [
                    "Problem Type",
                    PDFReportGenerator
                    ._pretty_name(
                        problem_type
                    )
                ],
                [
                    "Best Model",
                    str(
                        best_model
                    )
                ],
                [
                    "Selection Metric",
                    str(
                        selection_metric
                    )
                ],
                [
                    "Selection Score",
                    PDFReportGenerator
                    ._format_value(
                        selection_score
                    )
                ],
            ]

            automl_table = Table(
                automl_data,
                colWidths=[
                    2.5 * inch,
                    2.9 * inch
                ],
                repeatRows=1,
            )

            automl_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor(
                                "#17365D"
                            )
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold"
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                    ]
                )
            )

            elements.append(
                automl_table
            )

            # -------------------------------------------------
            # MODEL COMPARISON
            # -------------------------------------------------

            model_comparison = (
                automl_results.get(
                    "model_comparison"
                )
            )

            if model_comparison is not None:

                elements.append(
                    Spacer(
                        1,
                        0.2 * inch
                    )
                )

                elements.append(
                    Paragraph(
                        "7.1 Model Comparison",
                        subheading_style
                    )
                )

                try:

                    # Pandas DataFrame
                    if hasattr(
                        model_comparison,
                        "columns"
                    ):

                        comparison_columns = (
                            list(
                                model_comparison.columns
                            )
                        )

                        comparison_rows = (
                            model_comparison
                            .head(10)
                            .values
                            .tolist()
                        )

                        comparison_data = [
                            [
                                PDFReportGenerator
                                ._pretty_name(
                                    column
                                )
                                for column
                                in comparison_columns
                            ]
                        ]

                        for row in comparison_rows:

                            comparison_data.append(
                                [
                                    PDFReportGenerator
                                    ._format_value(
                                        value
                                    )
                                    for value
                                    in row
                                ]
                            )

                        available_width = (
                            7.0 * inch
                        )

                        column_width = (
                            available_width
                            / max(
                                len(
                                    comparison_columns
                                ),
                                1
                            )
                        )

                        comparison_table = Table(
                            comparison_data,
                            colWidths=[
                                column_width
                            ]
                            * len(
                                comparison_columns
                            ),
                            repeatRows=1,
                        )

                        comparison_table.setStyle(
                            TableStyle(
                                [
                                    (
                                        "BACKGROUND",
                                        (0, 0),
                                        (-1, 0),
                                        colors.HexColor(
                                            "#4F81BD"
                                        )
                                    ),
                                    (
                                        "TEXTCOLOR",
                                        (0, 0),
                                        (-1, 0),
                                        colors.white
                                    ),
                                    (
                                        "FONTNAME",
                                        (0, 0),
                                        (-1, 0),
                                        "Helvetica-Bold"
                                    ),
                                    (
                                        "GRID",
                                        (0, 0),
                                        (-1, -1),
                                        0.4,
                                        colors.grey
                                    ),
                                    (
                                        "FONTSIZE",
                                        (0, 0),
                                        (-1, -1),
                                        7
                                    ),
                                    (
                                        "VALIGN",
                                        (0, 0),
                                        (-1, -1),
                                        "MIDDLE"
                                    ),
                                ]
                            )
                        )

                        elements.append(
                            comparison_table
                        )

                except Exception:

                    elements.append(
                        Paragraph(
                            (
                                "Model comparison data "
                                "could not be formatted."
                            ),
                            body_style
                        )
                    )

        else:

            elements.append(
                Paragraph(
                    (
                        "AutoML was not trained before "
                        "this report was generated."
                    ),
                    body_style
                )
            )

        # =====================================================
        # 8. PREDICTION ANALYSIS
        # =====================================================

        elements.append(
            PageBreak()
        )

        elements.append(
            Paragraph(
                "8. Prediction Analysis",
                heading_style
            )
        )

        if prediction_result:

            predicted_value = (
                prediction_result.get(
                    "prediction",
                    "N/A"
                )
            )

            confidence = (
                prediction_result.get(
                    "confidence"
                )
            )

            prediction_data = [
                [
                    "Prediction Property",
                    "Result"
                ],
                [
                    "Target",
                    str(
                        automl_results.get(
                            "target_column",
                            "N/A"
                        )
                    )
                ],
                [
                    "Prediction",
                    str(
                        predicted_value
                    )
                ],
            ]

            if confidence is not None:

                try:

                    confidence_text = (
                        f"{float(confidence) * 100:.2f}%"
                    )

                except Exception:

                    confidence_text = str(
                        confidence
                    )

                prediction_data.append(
                    [
                        "Confidence",
                        confidence_text
                    ]
                )

            prediction_table = Table(
                prediction_data,
                colWidths=[
                    2.5 * inch,
                    2.9 * inch
                ],
                repeatRows=1,
            )

            prediction_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor(
                                "#17365D"
                            )
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold"
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                    ]
                )
            )

            elements.append(
                prediction_table
            )

            # -------------------------------------------------
            # CLASS PROBABILITIES
            # -------------------------------------------------

            probabilities = (
                prediction_result.get(
                    "class_probabilities"
                )
            )

            if probabilities:

                elements.append(
                    Spacer(
                        1,
                        0.2 * inch
                    )
                )

                elements.append(
                    Paragraph(
                        "8.1 Class Probabilities",
                        subheading_style
                    )
                )

                probability_data = [
                    [
                        "Class",
                        "Probability"
                    ]
                ]

                for (
                    class_name,
                    probability
                ) in probabilities.items():

                    try:

                        probability_text = (
                            f"{float(probability) * 100:.2f}%"
                        )

                    except Exception:

                        probability_text = str(
                            probability
                        )

                    probability_data.append(
                        [
                            str(
                                class_name
                            ),
                            probability_text
                        ]
                    )

                probability_table = Table(
                    probability_data,
                    colWidths=[
                        2.7 * inch,
                        2.7 * inch
                    ],
                    repeatRows=1,
                )

                probability_table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor(
                                    "#4F81BD"
                                )
                            ),
                            (
                                "TEXTCOLOR",
                                (0, 0),
                                (-1, 0),
                                colors.white
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.5,
                                colors.grey
                            ),
                        ]
                    )
                )

                elements.append(
                    probability_table
                )

            # -------------------------------------------------
            # PREDICTION INPUT
            # -------------------------------------------------

            if prediction_input:

                elements.append(
                    Spacer(
                        1,
                        0.2 * inch
                    )
                )

                elements.append(
                    Paragraph(
                        "8.2 Prediction Input Summary",
                        subheading_style
                    )
                )

                input_data = [
                    [
                        "Feature",
                        "Input Value"
                    ]
                ]

                for feature, value in list(
                    prediction_input.items()
                )[:30]:

                    input_data.append(
                        [
                            PDFReportGenerator
                            ._pretty_name(
                                feature
                            ),
                            html.escape(
                                str(value)
                            )
                        ]
                    )

                input_table = Table(
                    input_data,
                    colWidths=[
                        3.0 * inch,
                        2.4 * inch
                    ],
                    repeatRows=1,
                )

                input_table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor(
                                    "#D9EAF7"
                                )
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.4,
                                colors.grey
                            ),
                            (
                                "FONTSIZE",
                                (0, 0),
                                (-1, -1),
                                8
                            ),
                        ]
                    )
                )

                elements.append(
                    input_table
                )

        else:

            elements.append(
                Paragraph(
                    (
                        "No prediction had been generated "
                        "when this report was created."
                    ),
                    body_style
                )
            )

        # =====================================================
        # 9. PREDICTION EXPLAINABILITY
        # =====================================================

        elements.append(
            PageBreak()
        )

        elements.append(
            Paragraph(
                "9. Prediction Explainability",
                heading_style
            )
        )

        # -----------------------------------------------------
        # LOCAL EXPLANATION
        # -----------------------------------------------------

        if local_explanation.get(
            "available",
            False
        ):

            elements.append(
                Paragraph(
                    "9.1 Why This Prediction?",
                    subheading_style
                )
            )

            explained_class = (
                local_explanation.get(
                    "predicted_class",
                    "N/A"
                )
            )

            elements.append(
                Paragraph(
                    (
                        "The following contribution values "
                        "describe factors supporting or "
                        "opposing the model prediction for "
                        f"<b>{html.escape(str(explained_class))}</b>."
                    ),
                    body_style
                )
            )

            supporting = (
                local_explanation.get(
                    "increasing_factors",
                    []
                )
            )

            opposing = (
                local_explanation.get(
                    "decreasing_factors",
                    []
                )
            )

            if supporting:

                elements.append(
                    Paragraph(
                        "Factors Supporting the Prediction",
                        subheading_style
                    )
                )

                supporting_data = [
                    [
                        "Feature",
                        "Contribution"
                    ]
                ]

                for factor in supporting[:8]:

                    supporting_data.append(
                        [
                            str(
                                factor.get(
                                    "Feature",
                                    "Unknown"
                                )
                            ),

                            PDFReportGenerator
                            ._format_value(
                                factor.get(
                                    "Contribution"
                                )
                            )
                        ]
                    )

                supporting_table = Table(
                    supporting_data,
                    colWidths=[
                        3.5 * inch,
                        1.9 * inch
                    ],
                    repeatRows=1,
                )

                supporting_table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor(
                                    "#D9EAD3"
                                )
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.4,
                                colors.grey
                            ),
                        ]
                    )
                )

                elements.append(
                    supporting_table
                )

            if opposing:

                elements.append(
                    Spacer(
                        1,
                        0.15 * inch
                    )
                )

                elements.append(
                    Paragraph(
                        "Factors Opposing the Prediction",
                        subheading_style
                    )
                )

                opposing_data = [
                    [
                        "Feature",
                        "Contribution"
                    ]
                ]

                for factor in opposing[:8]:

                    opposing_data.append(
                        [
                            str(
                                factor.get(
                                    "Feature",
                                    "Unknown"
                                )
                            ),

                            PDFReportGenerator
                            ._format_value(
                                factor.get(
                                    "Contribution"
                                )
                            )
                        ]
                    )

                opposing_table = Table(
                    opposing_data,
                    colWidths=[
                        3.5 * inch,
                        1.9 * inch
                    ],
                    repeatRows=1,
                )

                opposing_table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor(
                                    "#F4CCCC"
                                )
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.4,
                                colors.grey
                            ),
                        ]
                    )
                )

                elements.append(
                    opposing_table
                )

        else:

            elements.append(
                Paragraph(
                    (
                        "A local prediction explanation "
                        "was not available."
                    ),
                    body_style
                )
            )

        # -----------------------------------------------------
        # GLOBAL IMPORTANCE
        # -----------------------------------------------------

        if global_explanation.get(
            "available",
            False
        ):

            elements.append(
                Spacer(
                    1,
                    0.2 * inch
                )
            )

            elements.append(
                Paragraph(
                    "9.2 Overall Model Importance",
                    subheading_style
                )
            )

            top_features = (
                global_explanation.get(
                    "top_features",
                    []
                )
            )

            if top_features:

                importance_data = [
                    [
                        "Feature",
                        "Importance",
                        "Relative Importance"
                    ]
                ]

                for feature in top_features[:10]:

                    importance_data.append(
                        [
                            str(
                                feature.get(
                                    "Feature",
                                    "Unknown"
                                )
                            ),

                            PDFReportGenerator
                            ._format_value(
                                feature.get(
                                    "Importance"
                                )
                            ),

                            (
                                f"{float(feature.get('Relative Importance (%)', 0)):.2f}%"
                            )
                        ]
                    )

                importance_table = Table(
                    importance_data,
                    colWidths=[
                        2.8 * inch,
                        1.3 * inch,
                        1.5 * inch
                    ],
                    repeatRows=1,
                )

                importance_table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor(
                                    "#4F81BD"
                                )
                            ),
                            (
                                "TEXTCOLOR",
                                (0, 0),
                                (-1, 0),
                                colors.white
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.4,
                                colors.grey
                            ),
                            (
                                "FONTSIZE",
                                (0, 0),
                                (-1, -1),
                                8
                            ),
                        ]
                    )
                )

                elements.append(
                    importance_table
                )

        # =====================================================
        # 10. AI BUSINESS EXPLANATION
        # =====================================================

        elements.append(
            PageBreak()
        )

        elements.append(
            Paragraph(
                "10. AI Business Explanation",
                heading_style
            )
        )

        if ai_business_explanation:

            elements.append(
                Paragraph(
                    PDFReportGenerator
                    ._clean_text(
                        ai_business_explanation
                    ),
                    body_style
                )
            )

        else:

            elements.append(
                Paragraph(
                    (
                        "An AI business explanation was not "
                        "generated before this report was created."
                    ),
                    body_style
                )
            )

        # =====================================================
        # 11. CONCLUSION
        # =====================================================

        elements.append(
            Paragraph(
                "11. Conclusion",
                heading_style
            )
        )

        elements.append(
            Paragraph(
                (
                    "The Enterprise AI Data Analyst Copilot "
                    "performed dataset profiling, data quality "
                    "assessment, exploratory analysis, AI-assisted "
                    "insight generation, machine learning "
                    "recommendation, automated model evaluation, "
                    "prediction, and model explainability where "
                    "the required outputs were available."
                ),
                body_style
            )
        )

        elements.append(
            Spacer(
                1,
                0.15 * inch
            )
        )

        elements.append(
            Paragraph(
                (
                    "<b>Important:</b> Machine-learning predictions "
                    "and AI-generated explanations should be "
                    "validated before they are used for operational, "
                    "employment, financial, safety-critical, or other "
                    "high-impact decisions."
                ),
                body_style
            )
        )

        elements.append(
            Spacer(
                1,
                0.2 * inch
            )
        )

        elements.append(
            Paragraph(
                (
                    "<i>Generated automatically by Enterprise "
                    "AI Data Analyst Copilot.</i>"
                ),
                small_style
            )
        )

        # =====================================================
        # BUILD
        # =====================================================

        doc.build(
            elements,
            onFirstPage=(
                PDFReportGenerator
                ._add_page_number
            ),
            onLaterPages=(
                PDFReportGenerator
                ._add_page_number
            ),
        )
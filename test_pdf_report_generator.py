from app.services.pdf_report_generator import PDFReportGenerator

profile = {
    "number_of_rows": 100,
    "number_of_columns": 8
}

cleaning = {
    "duplicates_removed": 5
}

eda = {
    "shape": (100, 8)
}

PDFReportGenerator.generate(
    "sample_report.pdf",
    profile,
    cleaning,
    eda,
    "This dataset looks clean and suitable for ML.",
    "Use Random Forest or XGBoost."
)

print("PDF generated successfully.")
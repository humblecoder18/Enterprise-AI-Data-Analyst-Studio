class ReportGenerator:
    """Generates a readable text report from dataset analysis."""

    @staticmethod
    def generate(profile: dict, cleaning_summary: dict, eda_results: dict) -> str:
        report = []

        report.append("=" * 50)
        report.append("Enterprise AI Data Analyst Report")
        report.append("=" * 50)

        report.append("\nDATASET SUMMARY")
        report.append("-" * 50)
        report.append(f"Rows: {profile['number_of_rows']}")
        report.append(f"Columns: {profile['number_of_columns']}")
        report.append(f"Numeric Columns: {len(profile['numeric_columns'])}")
        report.append(f"Categorical Columns: {len(profile['categorical_columns'])}")
        report.append(
            f"Missing Values: {sum(profile['missing_values_per_column'].values())}"
        )
        report.append(f"Duplicate Rows: {profile['duplicate_row_count']}")

        report.append("\nCLEANING SUMMARY")
        report.append("-" * 50)
        report.append(
            f"Duplicates Removed: {cleaning_summary['duplicates_removed']}"
        )
        report.append(
            f"Empty Rows Removed: {cleaning_summary['empty_rows_removed']}"
        )
        report.append(
            f"Columns Renamed: {cleaning_summary['columns_renamed']}"
        )

        report.append("\nEDA SUMMARY")
        report.append("-" * 50)
        report.append(f"Dataset Shape: {eda_results['shape']}")

        report.append("\nReport Generated Successfully.")

        return "\n".join(report)
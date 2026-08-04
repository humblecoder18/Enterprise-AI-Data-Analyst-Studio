from app.services.file_loader import FileLoader
from app.services.dataset_profiler import DatasetProfiler
from app.services.data_cleaning_agent import DataCleaningAgent
from app.services.eda_agent import EDAAgent
from app.services.report_generator import ReportGenerator
from app.services.visualization_agent import VisualizationAgent


def main():
    file_path = input("Enter CSV file path: ")

    # Load dataset
    df = FileLoader.load(file_path)
    print("\nDataset Loaded Successfully!")

    # Profile dataset
    profile = DatasetProfiler.profile(df)
    print("\nDataset Profile")
    print(profile)

    # Clean dataset
    cleaned_df, cleaning_summary = DataCleaningAgent.clean_data(df)
    print("\nCleaning Summary")
    print(cleaning_summary)

    #visualize dataset
    VisualizationAgent.generate_charts(cleaned_df)
    print("Charts generated successfully!")

    # EDA
    eda = EDAAgent(cleaned_df)
    eda_results = eda.analyze()
    print("\nEDA Completed")
    print(eda_results)

    # Generate report
    report = ReportGenerator.generate(
        profile,
        cleaning_summary,
        eda_results
    )

    print("\n")
    print(report)


if __name__ == "__main__":
    main()
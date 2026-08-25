import pandas as pd

from app.services.automl_engine import AutoMLEngine


# ---------------------------------------------------------
# Sample Dataset
# ---------------------------------------------------------

data = {
    "Age": [
        25, 32, 41, 29, 36,
        45, 27, 39, 31, 50,
        24, 34, 42, 28, 37,
        46, 26, 40, 33, 48
    ],

    "MonthlyIncome": [
        3000, 5000, 8000, 4200, 6200,
        9000, 3500, 7200, 4800, 10000,
        2800, 5500, 8200, 3900, 6500,
        9200, 3300, 7500, 5100, 9800
    ],

    "Department": [
        "Sales", "HR", "R&D", "Sales", "R&D",
        "HR", "Sales", "R&D", "HR", "R&D",
        "Sales", "HR", "R&D", "Sales", "R&D",
        "HR", "Sales", "R&D", "HR", "R&D"
    ],

    "OverTime": [
        "Yes", "No", "No", "Yes", "No",
        "Yes", "Yes", "No", "No", "Yes",
        "Yes", "No", "No", "Yes", "No",
        "Yes", "Yes", "No", "No", "Yes"
    ],

    "Attrition": [
        "Yes", "No", "No", "Yes", "No",
        "No", "Yes", "No", "No", "No",
        "Yes", "No", "No", "Yes", "No",
        "No", "Yes", "No", "No", "No"
    ]
}


df = pd.DataFrame(data)


# ---------------------------------------------------------
# AutoML Test
# ---------------------------------------------------------

engine = AutoMLEngine(
    df=df,
    target_column="Attrition"
)

result = engine.run()


print("\n==============================")
print("       AutoML RESULTS")
print("==============================")

print(
    "\nProblem Type:",
    result["problem_type"]
)

print(
    "Target Column:",
    result["target_column"]
)

print(
    "Best Model:",
    result["best_model"]
)

print("\nModel Comparison:\n")

print(
    result["results"].to_string(
        index=False
    )
)
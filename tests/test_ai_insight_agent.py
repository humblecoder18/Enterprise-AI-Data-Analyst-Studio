from app.services.ai_insight_agent import AIInsightAgent

sample_eda = {
    "shape": (100, 5),
    "missing_values": {
        "Age": 5,
        "Salary": 0
    },
    "summary_statistics": {
        "mean": {
            "Age": 31,
            "Salary": 52000
        }
    },
    "correlation_matrix": {
        "Age": {
            "Salary": 0.82
        }
    },
    "categorical_analysis": {
        "Department": {
            "unique_count": 4,
            "top_5_values": {
                "Sales": 40,
                "HR": 20
            }
        }
    }
}

agent = AIInsightAgent()

print(agent.generate_insights(sample_eda))
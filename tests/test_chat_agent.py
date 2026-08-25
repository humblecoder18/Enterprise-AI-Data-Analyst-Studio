import pandas as pd

from app.services.chat_agent import ChatAgent

df = pd.DataFrame({
    "Department": ["HR", "Sales", "HR", "IT"],
    "Salary": [30000, 50000, 35000, 70000]
})

agent = ChatAgent()

print(
    agent.ask(
        df,
        "Which department has the highest salary?"
    )
)
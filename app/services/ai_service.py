import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class AIService:
    """
    Shared Enterprise AI Service wrapping Groq.
    Loads GROQ_API_KEY from environment variables.
    """

    def __init__(self, model_name="llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def ask(self, prompt: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Enterprise AI "
                            "Data Analyst. Answer professionally "
                            "using the supplied dataset context."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq Error: {str(e)}")

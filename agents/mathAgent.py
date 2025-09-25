from google.adk.agents import LlmAgent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

# llm_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


def add_numbers(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


def power(base: float, exponent: float) -> float:
    """Raise base to the exponent"""
    return base ** exponent


# Create the LlmAgent
math_agent = LlmAgent(
    name="math_agent",
    instruction=(
        "You are a math expert agent. Your main goal is to solve math queries.\n\n"
        "INSTRUCTIONS:\n"
        "- Use multiply_numbers to multiply values.\n"
        "- Use add_numbers to add values.\n"
        "- Use power to compute exponentiation.\n"
        "- Use as many tools as needed depending on the user query.\n"
        "- Provide the final answer directly after computing."
    ),
    tools=[ multiply_numbers,add_numbers,power ],
    model="gemini-2.0-flash",
)


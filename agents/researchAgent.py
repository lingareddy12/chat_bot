from google.adk.agents import LlmAgent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

# llm_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

def get_product_price(product_name: str) -> float:
    """
    Retrieves the price of a product from an inventory.
    input: product_name (string) - The name of the product.
    """
    prices = {"pizza": 15.00, "soda": 2.50, "fries": 5.00}
    return prices.get(product_name.lower(), 0.00)


def calculate_total_cost(price: float, quantity: int, tax_rate: float = 0.10) -> float:
    """
    Calculates the total cost for a number of items, including a tax rate.
    input: price (float) - The price of a single item.
           quantity (int) - The number of items.
           tax_rate (float) - The tax rate (e.g., 0.10 for 10%).
    """
    subtotal = price * quantity
    tax_amount = subtotal * tax_rate
    return subtotal + tax_amount


# Create the LlmAgent
research_agent = LlmAgent(
    name="research_agent",
    instruction=(
       "You are an expert at handling prices and costs. Your main goal is to answer the user's question.\n\n"
        "use the available tool(s) to answer the question"
        "INSTRUCTIONS:\n"
        "- Use the get_product_price tool to get the price of an item first."
        "- Use the calculate_total_cost tool to compute the final cost, including a 10% tax."
        "- Give the final answer directly without any further modifications."
    ),
    tools=[get_product_price,calculate_total_cost],
    model="gemini-2.0-flash",
)


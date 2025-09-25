from mcp.server.fastmcp import FastMCP
from google.adk.tools.tool_context import ToolContext
import time

mcp = FastMCP("Custom Tools")

@mcp.tool()
def get_current_weather(city: str) -> str:
    """Returns the current weather for a specified city."""
    # (Simplified) API call to a weather service
    return f"The weather in {city} is sunny and 75 degrees."

@mcp.tool()
def generate_marketing_text() -> str:
    """A mock tool that returns a piece of generated text."""
    print("💡 The agent is generating marketing text...")
    time.sleep(1)  # Simulate processing
    return "Our new product is here! It's super fast and efficient. Get yours today!"

if __name__ == "__main__":
    mcp.run(transport="stdio")
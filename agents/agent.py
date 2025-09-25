# agent.py
# Copyright 2025 Google LLC
from google.adk import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from google.adk.tools.base_tool import BaseTool
import time
from typing import Dict, Any
from google.genai import types
from google.adk.tools.agent_tool import AgentTool
import uuid
import asyncio

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from .mathAgent import math_agent
# --- Tool Definitions ---

# mixed_toolset = MCPToolset(
#     connection_params=StdioConnectionParams(
#         server_params=StdioServerParameters(
#             command="python",
#             args=["server.py"]  
#         )
#     )
# )

def generate_marketing_text() -> str:
    """A mock tool that returns a piece of generated text."""
    print("💡 The agent is generating marketing text...")
    time.sleep(1)  # Simulate processing
    return "Our new product is here! It's super fast and efficient. Get yours today!"


def send_email(recipient: str, content: str, tool_context: ToolContext) -> dict:
    """
    Send an email. If confirmation is required, request it from the human-in-the-loop.
    """
    tool_confirmation = tool_context.tool_confirmation

    print("-------------------------------------36")
    print(tool_confirmation)
    print("--------------------------------38")


    if not tool_confirmation:
        tool_context.request_confirmation(
            hint="Please confirm whether to send the email with the provided content.",
            payload={
                "approved": False
            }
        )

        return {'status': 'Human approval is required.'}


    confirmed = tool_confirmation.payload['approved']

    if not confirmed:
        # User explicitly rejected
        return {"status": "rejected", "recipient": recipient, "content": content}

    success = True  
    return {"status": "ok" if success else "failed", "recipient": recipient, "content": content}









# --- Optional callback before every tool ---
def before_tool_callback(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    print("--------------------------------------------------89")
    print(f"[Callback] Tool '{tool.name}' started with args: {args}")
    print("--------------------------------------------------91")
    return None
def after_root_tool_callback(tool: BaseTool, args: Dict[str, Any],tool_response: Dict, tool_context: ToolContext):
    print("--------------------------------------------------94")
    print(f"[Callback] : Tool {tool.name} is completed with response : /n {tool_response} ")
    print("--------------------------------------------------96")
    return None

# --- Create the LlmAgent ---
root_agent = Agent(
    name="root_agent",
    instruction="""
    You are a helpful assistant that can help employees with creating marketing content and send emails
    - Use the `marketing_tool` tool for marketing requests.
    - Use the `send_email` tool for mail requests.
    - Use the  subagent `math_agent` for mathematics requests.
    - Prioritize using tools to fulfill the user's request.
    - - If an email requires human approval, do not continue with other tasks until approval is received.
    - Always respond to the user with the tool results.
    NOTE:
      - always return the result in a well structured way with proper new lines,points and spaces wherever needed
      - if the data is structured use tables , key : value pairs if needed
     """,
    tools=[#FunctionTool(generate_marketing_text,),
            AgentTool(agent=math_agent),
            send_email,generate_marketing_text],
    model="gemini-2.5-flash",
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_root_tool_callback,
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)

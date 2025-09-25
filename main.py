from fastapi import FastAPI,Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict,List,Optional
import httpx
import json

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or ["http://localhost:5500"] etc
    allow_credentials=True,
    allow_methods=["*"],        # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

ADK_URL = "http://localhost:8000/run_sse"

class ChatBody(BaseModel): 
    sessionId :str
    text: Optional[str] = None
    imgData: Optional[str] = None
    confirmationId: Optional[str] = None
    approvedValue: Optional[str] = None


ADK_URL_RUN = "http://localhost:8000/run_sse" # Replace with your actual ADK service URL



@app.post("/createSession")
async def create_session(sessionId: str = Body(..., embed=True)):
    """
    Creates a new session by forwarding the request to the ADK.
    """
    APP_NAME = "agents"
    USER_ID = "samp123"
    ADK_URL = f"http://localhost:8000/apps/{APP_NAME}/users/{USER_ID}/sessions/{sessionId}"
    
    payload = {
        "app_name": APP_NAME,
        "user_id": USER_ID,
        "session_id": sessionId
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(ADK_URL, json=payload)
            resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        return {"status": "success"}

    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code}")
        return {"status": "fail", "details": e.response.text}
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}.")
        return {"status": "fail", "details": str(e)}


@app.post("/chat")
async def chat_proxy(body: ChatBody):
    """
    Receives full ADK request from frontend,
    forwards it to /run_sse, and streams back only text and tool confirmation IDs.
    
    """
    async def event_stream():

        part_item: Dict[str, Any] = {}
        if body.text not in (None, ""):
            part_item["text"] =body.text

        if body.imgData not in (None, ""):
            part_item["inline_data"] = {
                "mime_type": "image/",
                "data": body.imgData
            }

        if body.confirmationId not in (None, "") and body.approvedValue not in (None, ""):
            part_item["function_response"] = {
                "id": body.confirmationId,
                "name": "adk_request_confirmation",
                "response": {
                    "response": json.dumps({
                        "confirmed": True,
                        "payload": {"approved": body.approvedValue}
                    })
                }
            }

     

        # Final payload with static fields and parts as a list
        payload = {
            "app_name": "agents",
            "user_id": "samp123",
            "session_id": body.sessionId,
            "new_message": {
                "role": "user",
                "parts": [part_item]  #  list of dicts
            }
        }


        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", ADK_URL_RUN, json=payload) as resp:
                async for chunk in resp.aiter_lines():
                    if not chunk or not chunk.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(chunk[len("data:"):].strip())
                        content = payload.get("content", {})
                        parts = content.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                yield json.dumps({
                                                    "type": "text",
                                                    "text": part["text"]
                                                }) + "\n\n"

                            # Function call
                            elif "functionCall" in part:
                                fc = part["functionCall"]

                                # If it's a confirmation request
                                if fc.get("name") == "adk_request_confirmation":
                                    original_call = fc.get("args", {}).get("originalFunctionCall", {})
                                    tool_confirmation = fc.get("args", {}).get("toolConfirmation", {})

                                    # Send both the confirmation ID and original tool info
                                    tool_confirmation_info = {
                                        "confirmation_id": fc.get("id"),  # THIS is the ID to use when sending approval
                                        # "original_tool_id": original_call.get("id"),
                                        "name": original_call.get("name"),
                                        # "args": original_call.get("args", {}),
                                        "hint": tool_confirmation.get("hint"),
                                    }  
                                    payload = {
                                            "type": "tool_confirmation",
                                            **tool_confirmation_info
                                        }

                                    yield json.dumps(payload) + "\n\n"

                                else:
                                    # Regular function call
                                    func_call_info = {
                                        # "id": fc.get("id"),
                                        "name": fc.get("name"),
                                        # "args": fc.get("args", {})
                                    }
                                    payload = {
                                            "type": "function_call",
                                            **func_call_info
                                        }
                                    yield json.dumps(payload) + "\n\n"

                            # Function response
                            elif "functionResponse" in part:
                                fr = part["functionResponse"]

                                payload = {
                                    "type": "function_response",
                                    # "response": {
                                        "name": fr.get("name")
                                    # }
                                }
                                yield json.dumps(payload) + "\n\n"
                                
                    except Exception as e:
                        print("Error parsing chunk:", e)
                        continue

    return StreamingResponse(event_stream(), media_type="text/event-stream",headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                })
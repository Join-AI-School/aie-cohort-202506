"""Add fastapi

uv add fastapi uvicorn
"""

#%%

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn

# Import your agent setup
from .agent import graph
from langchain_core.runnables import RunnableConfig

app = FastAPI()

# --------------------------
# Request/Response Models
# --------------------------

class Message(BaseModel):
    role: str  # 'human', 'ai', etc.
    content: str

class AgentRequest(BaseModel):
    messages: List[Message]
    user_id: str = "user_123"
    thread_id: str = "conversation_1"

class AgentResponse(BaseModel):
    # history: List[Message]
    response: str

# --------------------------
# API Route
# --------------------------

@app.post("/chat", response_model=AgentResponse)
async def chat(request: AgentRequest):
    # Convert to LangGraph's MessagesState format (list of tuples)
    messages: List[Tuple[str, str]] = [(msg.role, msg.content) for msg in request.messages]

    config = RunnableConfig(
        configurable={
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        },
        recursion_limit=25
    )

    # Invoke the LangGraph agent
    result = graph.invoke({"messages": messages}, config)

    print(result)

    # # Convert result["messages"] back to Pydantic-compatible format
    # history = [Message(role=m.role, content=m.content) for m in result["messages"]]

    # # Optional: Get just the final reply
    # reply = history[-1].content if history else "No response"

    # return AgentResponse(history=history, reply=reply)

    # Extract response message
    reply = result["messages"][-1].content if result and "messages" in result else "No response"

    return AgentResponse(response=reply)

# --------------------------
# Run the app (dev only)
# --------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
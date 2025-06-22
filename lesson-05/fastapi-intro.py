
"""Add fastapi

Install the libraries
```
uv add fastapi uvicorn
```

Run this from shell
```
uvicorn lesson-05.api:app --host 0.0.0.0 --port 8000 --reload
```
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request model for POST
class Message(BaseModel):
    name: str
    message: str

# GET endpoint
@app.get("/ping")
def ping():
    return {
        "message": "pong"
    }    

# POST endpoint
@app.post("/hello")
def say_hello(data: Message):
    return {
        "reply": f"Hello {data.name}, you said: {data.message}"
    }

# --------------------------
# Run the app (dev only)
# --------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8050, reload=True)
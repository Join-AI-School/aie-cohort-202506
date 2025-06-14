import os
import json
from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model_name="llama-3.3-70b-versatile",    
)

user_prompt = "How is the U.S. economy doing today? No more than 2 sentences."

response = model.invoke(user_prompt)
response = response.model_dump()
print(json.dumps(response, indent=4))

# model = init_chat_model(
#     "llama-3.1-8b-instant",
#     model_providers="groq"
# )

# model.inv
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# System Prompt
system_prompt_str = (
    "You are a financial agent, named 'FIA' Your task is to provide useful insights about the market and recomendations on personal finances."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        ("human", "{input}")
    ]
)
# print(prompt.invoke({"input": "What is your advice on investment?"}))

# LLM Initialization
model = ChatGroq(
    model_name="llama-3.3-70b-versatile",    
)

# Invoke chain
chain = prompt | model

response = chain.invoke("What is your advice on investment?")
response = response.model_dump()
print(json.dumps(response, indent=4))


response = chain.invoke("Should I invest in NVDA? And, what is your role?")
response = response.model_dump()
print(json.dumps(response, indent=4))
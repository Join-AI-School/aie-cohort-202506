#%%
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig, chain
from langchain.chat_models import init_chat_model
# For system prompt management
from langchain_core.prompts import ChatPromptTemplate
# Tavily Search API
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------------------------------------
# Set Environment Variables
# -----------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------
# Model
# -----------------------------------------------------------

model = init_chat_model(
    "llama3-70b-8192", 
    model_provider="groq",
    temperature=0,
)

# -----------------------------------------------------------
# System Prompt
# -----------------------------------------------------------

system_prompt_str = (
    "You are a financial insights advisor, named 'FIA'. Your "
    "role is to provide accurate and insightful financial "
    "recommendations based on the data provided. You should "
    "consider various financial metrics and market conditions "
    "to give the best advice possible. Always ensure your "
    "recommendations are clear, concise, and actionable. "
)

messages = [    
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        ("placeholder", "{messages}"),
        ("human", "{input}"),
    ]
)

prompt.invoke({"input": "How is the economy doing?", "messages": messages}).to_messages()

# -----------------------------------------------------------
# Set Tool
# -----------------------------------------------------------

tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
)

tool_msgs = tool.invoke("Economy")

#%%
# -----------------------------------------------------------
# Chain
# -----------------------------------------------------------

# LLM binded with tool. This tool will ALWAYS be called.
model_with_tool = model.bind_tools([tool])
llm_chain = prompt | model_with_tool

@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config) # Calling tool
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)

print(tool_chain.invoke("How's the economy doing today?").model_dump())

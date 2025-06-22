"""

Agent
"""

#%%
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# For system prompt management
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Memory persistence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph

# For system prompt management
from langchain_core.prompts import ChatPromptTemplate

# LangGraph Agent
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition

# Import custom chains
from tools.sql_agent import sql_agent_executor
from tools.vector_chain import vector_search

# Display graph
from IPython.display import Image, display

# -----------------------------------------------------------
# Set Environment
# -----------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------
# Model
# -----------------------------------------------------------

# model = init_chat_model(
#     "gpt-4o",                  # Or "llama-3.3-70b-versatile", 
#     model_provider="openai",   # Or "groq"
#     temperature=0,
#     verbose=True
# )

model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0,
    verbose=True
)

# -----------------------------------------------------------
# Agent Tools
# -----------------------------------------------------------

@tool
def sql_agent_tool(query: str):
    """Wrapper to use SQL agent as a tool"""
    print(query)
    result = sql_agent_executor.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    print(result)
    return result['messages'][-1].content  # Return the final response

@tool
def tavily_search_tool(query: str):
    """Use this tool to search the web"""
    result = TavilySearchResults(
        max_results=1,
        search_depth="advanced",
        include_answer=True,
    )    
    return result.invoke({"query": query})[0]["content"]

@tool
def vector_search_tool(query: str):
    """Use this tool to get the latest news"""
    result = vector_search(query)
    return result

# Declare tools
tools = [tavily_search_tool, sql_agent_tool, vector_search_tool]
tool_node = ToolNode(tools)

# Bind tools to the LLM
model_with_tool = model.bind_tools(tools)

# -----------------------------------------------------------
# System Prompt
# -----------------------------------------------------------

system_prompt_str = (
"""
You are a financial insights advisor, named 'FIA'. Your
role is to provide accurate and insightful financial
recommendations based on the data provided.

REQUIREMENTS:
- Consider various financial metrics and market conditions to give the best advice possible.
- Always ensure your recommendations are clear, concise, and actionable
- Use available tools when they can help answer the query

TOOLS:
- `sql_agent_tool`: Use this to fetch the latest information on stock prices
- `tavily_search_tool`: use this to search the web
"""
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        ("placeholder", "{messages}"),
    ]
)

llm_chain = (prompt_template | model_with_tool)

# -----------------------------------------------------------
# Call the Model
# -----------------------------------------------------------

def call_model(state: MessagesState, config: RunnableConfig):
    response = llm_chain.invoke(state)
    return {"messages": response}

# -----------------------------------------------------------
# Create ReACT agent with Graph Nodes
# -----------------------------------------------------------

# Create memory saver()
memory = MemorySaver()

# Define a new graph
agent_graph = StateGraph(state_schema=MessagesState)

# Add graph nodes
agent_graph.add_node("model", call_model)
agent_graph.add_node("tools", tool_node)

# Add edges
agent_graph.add_edge(START, "model")
agent_graph.add_conditional_edges("model", tools_condition, ["tools", END])
agent_graph.add_edge("tools", "model")

# Compile the graph
graph = agent_graph.compile(checkpointer=memory)
# display(Image(graph.get_graph().draw_mermaid_png()))

# -----------------------------------------------------------
# Test Agent
# -----------------------------------------------------------

# for question in questions:
config = {
    "configurable": {
        "thread_id": "conversation_1",
        "user_id": "user_123"
    },
    "recursion_limit": 25 # Overwrites the default recursion of 25.
}

response = graph.invoke({
    "messages": [
        ("human", "What is the latest stock price on MSFT? Use tavily search")
    ]
}, config)

response




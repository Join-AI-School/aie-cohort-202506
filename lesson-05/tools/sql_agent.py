""" SQL Agent """
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# For system prompt management
from langchain import hub
# SQL DB Connection
from sqlalchemy.engine import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# LangGraph Agent
from langgraph.prebuilt import create_react_agent

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")  # e.g., "my-gcp-project"
GCP_BQ_DATASET = os.getenv("GCP_BQ_DATASET")  # default dataset name
GCP_BQ_TABLE = os.getenv("GCP_BQ_TABLE")  # default table name

service_account_file = "../service-account.json"

# -----------------------------------------------------------
# Model
# -----------------------------------------------------------

sql_model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0,
    verbose=True
)

# sql_model = init_chat_model(
#     "gpt-4o",
#     model_provider="openai",
#     temperature=0,
#     verbose=True
# )

# -----------------------------------------------------------
# SQL Tools
# -----------------------------------------------------------

# Connection URL
connection_url = f"bigquery://{GCP_PROJECT_ID}/{GCP_BQ_DATASET}"

# Create SQL Alchemy Engine
engine = create_engine(connection_url, credentials_path=service_account_file)

# Create the SQLDatabase object
db = SQLDatabase(engine)

# Verify that connection works
toolkit = SQLDatabaseToolkit(db=db, llm=sql_model)
tools = toolkit.get_tools()

# Set SQL Agent Prompt
prompt_template = (
    hub.pull("langchain-ai/sql-agent-system-prompt")
        .format(dialect="GCP BigQuery", top_k=5)
)

# Create ReACT agent with specified LLM, SQL tool & Prompt
sql_agent_executor = create_react_agent(
    sql_model, 
    tools, 
    prompt=prompt_template,
)

# # Invoke
# user_question = "How is the market performance latey?"
# response = sql_agent_executor.invoke({
#     "messages": [
#         ("user", user_question)
#     ]
# }); 

# print(response['messages'][-1].content)
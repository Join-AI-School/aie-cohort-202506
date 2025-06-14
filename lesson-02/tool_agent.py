from langchain_ollama.llms import OllamaLLM
import re

# Initialize Ollama Model
model = OllamaLLM(model="gemma3:latest")

user_task = "Perform this calculation 3 and 3"

system_prompt = f"""You are an assistant that has access to
two tools: calculator and language translator. 

Discern which tool you will need to use for the task. 
If you don't need any tool to perform task, just say 'None'.

- For calculation, explicitely call the tool like this:
    TOOL CALL: calculator("expression")
- For translation: call the tool like this:
    TOOL CALL: translator("text", "target_language")

*If you are using a calculator. Note that the user may provide an expression
that is not correct in terms syntax. You have to infer and correct it.

If you don't understand the user's question, just say 'I don't know.'

Here's your task:
{user_task}

Your response: 
"""

tool_response = model.invoke(system_prompt)
# print(tool_response)

# Actual tool functions
def calculator(expression):
    try:
        return eval(expression)
    except Exception as e:
        return str(e)    

def translation():
    print("perform translation!")

tool_call_pattern = r'TOOL CALL: (calculator|translator)\("(.+?)"(?:,\s*"(.+?)")?\)'
match = re.search(tool_call_pattern, tool_response)

if match:
    tool_name, arg1, arg2 = match.group(1), match.group(2), match.group(3)
    print(f"Extracted Tool Call: {tool_name}({arg1}{', ' + arg2 if arg2 else ''})")

    # Emulate tool execution
    if tool_name == "calculator":
        tool_result = calculator(arg1)

    print("TOOL RESPONSE:", tool_result)
from langchain_ollama.llms import OllamaLLM

# Initialize Ollama Model
model = OllamaLLM(model="gemma3:1b")

# -----------------------------------------------------------
# Store Memory
# -----------------------------------------------------------

# Initialize conversation memory
conversation_memory = []

while True: 
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user input to conversation memory
    conversation_memory.append(f"User: {user_input}")

    conversation_prompt = "\n".join(conversation_memory) + "\nAgent:"

    agent_response = model.invoke(conversation_prompt)
    response_text = agent_response.strip()

    print("Agent:", response_text)
    conversation_memory.append(f"Agent: {response_text}")
    
from langchain_ollama.llms import OllamaLLM

# Reflective Agent

# S1 - Generate the inital response
# S2 - Reflect on the quality 
# 

# Initialize Ollama Model
model = OllamaLLM(model="gemma3:1b")


prompt = "write me an essay about michael jordan."
initial_response = model.invoke(prompt)

# 
system_prompt = f"""
Based on the user prompt, provide a response on whether this is good enough. Be critical,
but only provide final PASS or FAIL in the response.

prompt: {prompt}
response: {initial_response}

Your decision: (PASS/FAIL)
"""

evaluation_response = model.invoke(system_prompt).strip()

# print(evaluation_response, evaluation_response=="PASS")

if evaluation_response == "PASS":
    print(initial_response)
else:
    evaluation_prompt = f"""
Based on the user prompt, provide a critique of what is lacking in this essay.

prompt: {prompt}
response: {initial_response}

Your feedback: 
"""
    feedback = model.invoke(evaluation_prompt)

    # print(feedback)

    refined_prompt = f"""
Based on the user prompt, initial_draft, feedback, provide a refined draft.

prompt: {prompt}
initial_draft: {initial_response}
feedback: {feedback}

Your final draft:
"""
    final = model.invoke(refined_prompt)

    print("---------------------------------------")
    print("Initial Response")
    print(initial_response)

    print("---------------------------------------")
    print("Final Response")
    print(final)

    
from agentic_rag.agent_flow import flow, MyState

# Example usage: run the RAG workflow for a sample query
state = MyState()
state["query"] = "Explain llm"
response = flow.invoke(state)
print("Final Answer:", response["response"])

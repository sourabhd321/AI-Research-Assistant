from agentic_rag.agent_flow import flow, MyState

import warnings

# Suppress specific deprecation/warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Convert_system_message_to_human will be deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated")


# Example usage: run the RAG workflow for a sample query
state = MyState()
state["query"] = "What is Blackhole"
response = flow.invoke(state)
print("Final Answer:", response["response"])

from langgraph.graph import StateGraph, START, END
from agentic_rag.tools_agent import agent_executor
from agentic_rag.state import MyState
from agentic_rag.refinement import refined_query_generator

# --- Query refinement node ---
def refinement_node(state: MyState):
    return refined_query_generator(state)


def agentic_rag_node(state: MyState):
    """
    A node that uses the tool-enabled agent to answer the user query.
    """
    prompt_text = (
    "You are a Retrieval-Augmented Generation (RAG) agent.\n"
    "Use the tools to:\n"
    "- Refine the query if needed,\n"
    "- Retrieve relevant documents from the vector DB,\n"
    "- If the knowledge base does not have enough info (relevance < 0.7), use the `search_web` tool to fetch web snippets,\n"
    "- Then use `summarize_text` to synthesize those snippets into a coherent explanation,\n"
    "- Score the relevance of the results, and\n"
    "- Generate a final answer.\n\n"
    "- Only output an answer if the relevance score is ≥ 0.7 on internal docs. If not, fall back:\n"
        "  1) call `search_web(query)`,\n"
        "  2) call `summarize_text` on the returned snippets,\n"
        "  3) then `generate_answer` using that summary as the document.\n"
    "Retry up to 5 times if necessary.\n\n"
    f"User query: {state['query']}"
)

    # Run the agent with the constructed prompt
    # result = agent_executor.invoke({"input": prompt_text})
    # # The agent returns a dict with an 'output' key containing the answer
    # return {"response": result["output"]}
    result = agent_executor.invoke({"input": prompt_text})

    score = ""
    if "intermediate_steps" in result:
        for action, observation in result["intermediate_steps"]:
            # action is the tool call; observation is the result (what your tool returned)
            if getattr(action, "tool", None) == "score_relevance":
                score = observation
    return {
        "response": result["output"],
        "refined_query": state.get("refined_query", ""),
        "score": score
    }

# Build the StateGraph for the workflow (single-node agentic approach)
graph = StateGraph(MyState)
graph.add_node("refinement", refinement_node)
graph.add_node("agentic", agentic_rag_node)
graph.add_edge(START, "refinement")
graph.add_edge("refinement", "agentic")
graph.add_edge("agentic", END)
flow = graph.compile()
print(flow)
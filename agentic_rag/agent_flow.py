from langgraph.graph import StateGraph, START, END
from agentic_rag.tools_agent import (
    agent_executor,
    search_web,
    generate_answer,   
)
from agentic_rag.state import MyState
from agentic_rag.refinement import refined_query_generator

# --- Query refinement node ---
def refinement_node(state: MyState):
    return refined_query_generator(state)


def agentic_rag_node(state: MyState):
    """
    A node that uses the tool-enabled agent to answer the user query,
    falling back to web search if vector-DB relevance is low.
    """
    # 1) Normalize & build the prompt for your tool-calling agent
    normalized_query = state["query"].replace("-", " ").strip()
    prompt_text = (
        "You are a Retrieval-Augmented Generation (RAG) agent.\n"
        "Use the tools to:\n"
        "- Refine the query,\n"
        "- Retrieve relevant documents from the vector DB,\n"
        "- If the knowledge base does not have enough info (relevance < 0.5), use `search_web`,\n"
        "- Finally, generate an answer.\n\n"
        f"User query: {normalized_query}"
    )

    # 2) Invoke the agent
    result = agent_executor.invoke({"input": prompt_text})

    # 3) Pull out the relevance score
    score = 0.0
    for action, observation in result.get("intermediate_steps", []):
        if getattr(action, "tool", None) == "score_relevance":
            try:
                score = float(observation)
            except:
                score = 0.0

    # ─── 4) FALLBACK: If internal score is low, go to web search ────────────────────
    if score < 0.5:
        raw_snippets  = search_web.invoke({"query": normalized_query})
        direct_answer = generate_answer.invoke({
            "query":    normalized_query,
            "document": raw_snippets,
        })
        return {
            "response":      direct_answer,
            "refined_query": state.get("refined_query", ""),
            "score":         score,
            "used_fallback": True,
        }

    # ─── 5) Otherwise, return the agent’s original RAG answer ────────────────────────
    return {
        "response":      result.get("output", ""),
        "refined_query": state.get("refined_query", ""),
        "score":         score,
        "used_fallback": False,
    }


# Build the StateGraph for the workflow
graph = StateGraph(MyState)
graph.add_node("refinement", refinement_node)
graph.add_node("agentic", agentic_rag_node)
graph.add_edge(START, "refinement")
graph.add_edge("refinement", "agentic")
graph.add_edge("agentic", END)
flow = graph.compile()

print(flow)

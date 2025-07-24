from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agentic_rag.llm_config import llm
from agentic_rag.state import MyState

class QueryRefiner(BaseModel):
    refined_query: str = Field(description='Give a refined query that helps retrieve relevant documents from the vector DB.')

# Prompt template for refining the user's query
refine_query_template = """You are an AI assistant that helps improve search performance by rewriting user queries into more detailed and precise versions, while preserving the original intent.

Refine the query by:
- Making the language more formal or descriptive
- Adding context or clarifying ambiguous terms if needed
- Including relevant keywords that would help retrieve the most relevant documents from a knowledge base

---

Original User Query:
{query}

Refined Query (return only the improved query):
"""
# Create a structured-output chain for query refinement
llm_refiner = llm.with_structured_output(QueryRefiner)
refine_query_prompt = PromptTemplate.from_template(refine_query_template)
refine_query_chain = refine_query_prompt | llm_refiner

def refined_query_generator(state: MyState):
    query = state["query"]
    count = int(state.get("count", 0)) + 1
    refined_obj = refine_query_chain.invoke({"query": query})
    if hasattr(refined_obj, "refined_query"):
        refined = refined_obj.refined_query
    elif isinstance(refined_obj, dict):
        refined = refined_obj.get("refined_query", "")
    else:
        refined = str(refined_obj)
    state["refined_query"] = refined
    state["count"] = str(count)
    return state

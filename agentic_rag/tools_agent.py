from langchain.tools import tool
from agentic_rag.retrieval import hybrid_retriever
from agentic_rag.refinement import refine_query_chain
from agentic_rag.scoring import scorer_chain
from agentic_rag.generation import generator_chain
from agentic_rag.llm_config import llm
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

def extract_field(result, field: str):
    # Handles BaseModel, dict, or message objects
    if hasattr(result, field):
        return getattr(result, field)
    if isinstance(result, dict):
        return result.get(field, "")
    # Sometimes result could be a message object with .content
    if hasattr(result, "content"):
        return getattr(result, "content")
    return str(result)

class SummarizationOutput(BaseModel):
    summary: str = Field(description="Concise summary of the input text")

summarizer_template = """You are an expert AI assistant. Summarize the following text into a clear, cohesive explanation suitable for answering a user’s question:\n\nText:\n{text}\n\nSummary:"""
summarizer_prompt = PromptTemplate.from_template(summarizer_template)
summarizer_chain = summarizer_prompt | llm.with_structured_output(SummarizationOutput)

@tool
def refine_query(query: str) -> str:
    """Refine the user query for better retrieval."""
    result = refine_query_chain.invoke({"query": query})
    return extract_field(result, "refined_query")

@tool
def retrieve_docs(query: str) -> str:
    """Retrieve documents from the vector store based on the (refined) query."""
    docs = hybrid_retriever.invoke(query)
    return "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])

@tool
def score_relevance(input_text: str) -> str:
    """Score the relevance of retrieved documents to the user's query. Returns a float between 0 and 1."""
    try:
        query_part, docs_part = input_text.split("Retrieved documents:", 1)
    except ValueError:
        query_part, docs_part = "", input_text

    query = query_part.strip()
    documents = docs_part.strip()
    result = scorer_chain.invoke({"query": query, "documents": documents})
    print("SCORE TOOL RAW RESULT:", result)
    if hasattr(result, "score"):
        print("SCORE TOOL EXTRACTED .score:", result.score)
        return result.score
    if isinstance(result, dict) and "score" in result:
        print("SCORE TOOL EXTRACTED ['score']:", result["score"])
        return result["score"]
    print("SCORE TOOL FALLBACK:", str(result))
    return str(result)

@tool
def search_web(query: str) -> str:
    """Search the web for relevant information using DuckDuckGo."""
    ddg = DuckDuckGoSearchAPIWrapper()
    results = ddg.run(query)
    return results

@tool
def summarize_text(text: str) -> str:
    """Summarize a block of text into a concise explanation."""
    result = summarizer_chain.invoke({"text": text})
    # # extract the summary field
    # return getattr(result, "summary", result.get("summary", str(result)))
    # the Pydantic model has a `.summary` attribute – return it directly
    if hasattr(result, "summary"):
        return result.summary
    # fallback if something unexpected happens
    return str(result)


@tool
def generate_answer(query: str, document: str) -> str:
    """Generate an answer to the user query using the provided document context."""
    result = generator_chain.invoke({"query": query, "document": document})
    # In some cases, result might be a message object or dict, so extract content
    return extract_field(result, "response") or str(result)

# Agent setup remains unchanged
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

tools = [generate_answer, score_relevance, retrieve_docs, refine_query, search_web, summarize_text]

tool_list_text = "\n".join(f"- `{tool.name}`: {tool.description}" for tool in tools)
system_message = (
    "You are a Retrieval-Augmented Generation (RAG) agent.\n"
    "Use your tools wisely to refine queries, retrieve documents, score relevance, and generate an answer.\n\n"
    "Available tools:\n"
    f"{tool_list_text}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", "{input}"),
    (MessagesPlaceholder(variable_name="agent_scratchpad", optional=True))
])

agent_runnable = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=False)

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agentic_rag.llm_config import llm

class RetrieverRelevance(BaseModel):
    score: str = Field(description='Relevance score between 0 and 1')

# Prompt template for evaluating relevance of documents to the query
check_relevant_template = """You are an AI assistant that evaluates how relevant a given document is in helping answer a user’s query. Relevance is scored between 0 and 1, where:

- 0 means the document is **not relevant at all**
- 1 means the document is **highly relevant and directly helpful** in answering the query

Carefully read the **user's query** and the **document content**, then return only the relevance score as a float between 0 and 1.

---

User Query:
{query}

Retrieved Document:
{documents}

---

Relevance Score (between 0 and 1):
"""
# Create a structured-output chain for relevance scoring
llm_scorer = llm.with_structured_output(RetrieverRelevance)
check_relevant_prompt = PromptTemplate.from_template(check_relevant_template)
scorer_chain = check_relevant_prompt | llm_scorer
print(scorer_chain)
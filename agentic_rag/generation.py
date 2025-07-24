from langchain_core.prompts import PromptTemplate
from agentic_rag.llm_config import llm

# Prompt template for generating the final answer from the retrieved document context
generator_template = """You are an intelligent assistant that answers user queries based solely on the information provided in the given document context.

Instructions:
- Read the document carefully.
- Answer the question accurately using only the information from the document.
- If the document does not contain enough information to answer the question, say:
  "The provided document does not contain enough information to answer this question."
- Do not include reasoning or explanation unless explicitly asked.

---

Document Context:
{document}

User Query:
{query}

---

Answer:
"""
generator_prompt = PromptTemplate.from_template(generator_template)
generator_chain = generator_prompt | llm

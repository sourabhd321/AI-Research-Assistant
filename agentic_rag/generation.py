from langchain_core.prompts import PromptTemplate
from agentic_rag.llm_config import llm

# Prompt template for generating the final answer from the retrieved document context
generator_template = """You are an intelligent assistant that answers user queries based solely on the information provided in the given document context.

Instructions:
- Treat the input “{document}” as your sole context (it may be from internal docs or web search).
- If the context answers the question, answer directly—do not mention where it came from.
- If it still doesn’t, reply: “I’m sorry, I couldn’t find an answer.”
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
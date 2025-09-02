# AI Research Assistant

## 📖 Overview
The **AI Research Assistant** is an agentic RAG (Retrieval-Augmented Generation) pipeline designed to support advanced research tasks.  
It combines **retrieval, reasoning, and generation** with modular components for ingestion, query refinement, and output generation.  
The system uses **LangChain**, **LangGraph**, and **FAISS**, enhanced with **BM25 hybrid retrieval** and **cross-encoder reranking** for high-precision results.  

On top of the pipeline, a **Streamlit web app** (`rag_streamlit_app.py`) provides an intuitive interface for researchers to interact with the assistant, ask questions, and visualize responses.

---

## 🚀 Key Features
- **Hybrid Retrieval**: Combines dense embeddings with BM25 for balanced semantic + keyword search.  
- **Reranking**: Cross-encoder model refines retrieved passages for better relevance.  
- **Agentic Reasoning**: Powered by LangGraph with conditional edges and iterative loops.  
- **Query Refinement**: Uses Gemini (via `langchain_google_genai`) to re-formulate queries for better coverage.  
- **Self-Critique & Improvement**: The pipeline can score and revise its own outputs.  
- **Modular Design**: Clean separation of ingestion, retrieval, refinement, generation, and agent flow.  
- **Streamlit App**: Modern UI for asking questions, viewing retrieved sources, and getting citation-backed answers.  

---

## 📂 Project Structure

ai_research_assistant/
│── agentic_rag/
│ │── agent_flow.py # Defines LangGraph state machine and agent workflow
│ │── ddg_check.py # DuckDuckGo search helper for external knowledge retrieval
│ │── generation.py # Handles LLM response generation using Gemini
│ │── ingestion.py # Ingests documents and builds FAISS vector store
│ │── init.py # Marks directory as a Python package
│ │── llm_config.py # Centralized config for Gemini and other LLMs
│ │── refinement.py # Query reformulation and refinement logic
│ │── retrieval.py # Hybrid (BM25 + dense) retrieval with reranking
│ │── scoring.py # Implements self-critique and scoring of answers
│ │── state.py # LangGraph state definitions and transitions
│ │── tools_agent.py # Utility tools exposed to the agent (search, calculators, etc.)
│
│── main.py # CLI entry point for running pipeline
│── rag_streamlit_app.py # Streamlit web app for interactive research assistant
│── .gitignore # Git ignore rules
│── README.md # Project documentation


---

## ⚙️ Tech Stack
- **Python 3.10+**  
- **LangChain** + **LangGraph**  
- **Vector Stores**: FAISS (with BM25 hybrid)  
- **LLMs**: Google Gemini via `langchain_google_genai`  
- **Rerankers**: Cross-encoders for document relevance  
- **Frontend**: Streamlit  

---

## 🛠️ Setup

1. Clone the repo:  
   ```bash
   git clone https://github.com/sourabhd321/ai-research-assistant.git
   cd ai-research-assistant

Create and activate virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

GOOGLE_API_KEY=your_gemini_key

▶️ Usage

Run the pipeline (CLI mode)
python main.py

Run the Streamlit app (UI mode)
streamlit run rag_streamlit_app.py


🏗️ Architecture

flowchart TD
    A[📂 Ingestion] -->|Load docs, build FAISS| B[🔍 Retrieval]
    B -->|Hybrid: BM25 + Dense| C[⚖️ Reranking]
    C --> D[🔧 Refinement]
    D --> E[🤖 Agent Flow (LangGraph)]
    E --> F[✍️ Generation (LLM - Gemini)]
    F --> G[⭐ Scoring & Self-Critique]
    G --> H[🎯 Final Answer]

    %% UI paths
    H -->|Display| I[🖥️ Streamlit App]
    E -->|CLI mode| J[💻 main.py]

## Authors

- [@sourabhd321](https://github.com/sourabhd321)


## Badges

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-Flan--T5-blueviolet)


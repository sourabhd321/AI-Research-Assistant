import streamlit as st
from agentic_rag.ingestion import RecursiveCharacterTextSplitter, HuggingFaceEmbeddings
from agentic_rag.state import MyState
from agentic_rag.agent_flow import flow
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

# --------- Function to load docs and track failures ----------
def load_docs_with_reporting(urls):
    docs = []
    failed_urls = []
    for url in urls:
        try:
            loader = UnstructuredURLLoader(urls=[url])
            loaded = loader.load()
            if not loaded or len(loaded) == 0:
                failed_urls.append(url)
            else:
                docs.extend(loaded)
        except Exception:
            failed_urls.append(url)
    return docs, failed_urls

# --------- Sidebar: User provides URLs ----------
st.sidebar.header("Document URLs")
urls_input = st.sidebar.text_area(
    "Enter one URL per line:",
    height=150,
    value=""
)

if st.sidebar.button("Load Documents"):
    urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
    st.session_state['urls'] = urls

    with st.spinner("Loading and chunking documents..."):
        docs, failed_urls = load_docs_with_reporting(urls)
        st.session_state['docs'] = docs

        # Chunking and embedding
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        st.session_state['db'] = db

    st.success(f"Documents loaded and vector store created! Loaded {len(docs)} documents.")

    # ---- Display failed URLs if any ----
    if failed_urls:
        st.error("Some URLs could not be processed:")
        for url in failed_urls:
            st.markdown(f"- {url}")

# --------- Main Panel: Question Answering ----------
st.header("Ask a Question")

if 'db' in st.session_state:
    user_query = st.text_input("Your question:")

    if st.button("Get Answer") and user_query:
        # Patch: dynamically replace your vector DB in the pipeline
        import agentic_rag.ingestion  # noqa: F401
        agentic_rag.ingestion.vector_db = st.session_state['db']

        state = MyState()
        state["query"] = user_query

        with st.spinner("Thinking..."):
            response = flow.invoke(state)

        # ---- Show Refined Query (if present) ----
        refined_query = response.get("refined_query", "")
        if refined_query:
            st.markdown("**Refined Question:**")
            st.code(refined_query, language="text")

        # ---- Show Final Answer ----
        st.markdown("**Final Answer:**")
        st.info(response.get("response", ""))

        # ---- Display Source Note ----
        used_fallback = response.get("used_fallback", False)
        if used_fallback:
            st.warning("*Note:* This answer was sourced from Tavily search.")
        else:
            st.success("*Note:* This answer was sourced from the vector database.")

else:
    st.warning("Please enter URLs and load documents first.")

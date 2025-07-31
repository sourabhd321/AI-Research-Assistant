from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from agentic_rag.ingestion import vector_db, chunks
from agentic_rag.state import MyState
from langchain.retrievers import EnsembleRetriever

# Set up dense and sparse retrievers
dense_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
sparse_retriever = BM25Retriever.from_documents(chunks, k1=1.5, b=0.75)

# Combine dense and sparse retrievers with equal weights for a hybrid approach
hybrid_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5])

# Initialize a cross-encoder model for reranking retrieved documents
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def final_retriever(STATE: MyState):
    """Retrieve relevant documents for the query in STATE and rerank them."""
    query = STATE['query']
    # Retrieve documents using the hybrid retriever
    docs = hybrid_retriever.get_relevant_documents(query)
    # Prepare query-document pairs for reranking
    pairs = [[query, doc.page_content] for doc in docs]
    # Compute relevance scores for each pair using the cross-encoder
    scores = reranker_model.predict(pairs)
    # Sort documents by score (highest score first)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    # Combine the content of the top 5 documents into one string
    top_docs_content = "\n\n".join([doc.page_content for doc, _ in scored_docs[:5]])
    return {"document": top_docs_content}
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Define the URLs to load documents from
urls = [
    "https://generativeai.net/",
    "https://huggingface.co/learn/cookbook/en/agent_rag",
    "https://www.linkedin.com/pulse/generative-ai-vs-llm-what-big-difference-techmobius-6o6lc"
]

# Initialize the loader and load documents from the URLs
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Split documents into smaller chunks for processing
chunk_size = 1000
chunk_overlap = 200
chunkingsplitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap, 
    separators=["\n\n", "\n", " ", ""]
)
chunks = chunkingsplitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Initialize HuggingFace embeddings model and create a FAISS vector store from document chunks
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, hf_embeddings)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()
# Sample documents
docs2 = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
# Initialize Huggingface embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Create the FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=docs2,
    embedding=embedding_model
)
# Enable MMR in the retriever
retriever2 = vectorstore.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 2, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)
query = "What is langchain?"
results2 = retriever2.invoke(query)

for i, doc in enumerate(results2):
    print(f"\n--- MMR Result {i+1} ---")
    print(doc.page_content)